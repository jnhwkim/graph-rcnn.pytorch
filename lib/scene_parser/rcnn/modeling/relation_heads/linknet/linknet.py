# LinkNet model (Woo et al., 2018)
# Reimplemented by Jin-Hwa Kim (jnhwkim@sktbrain.com)
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...roi_heads.box_head.roi_box_predictors import make_roi_box_predictor
from ..roi_relation_feature_extractors import make_roi_relation_feature_extractor
from ..roi_relation_box_feature_extractors import make_roi_relation_box_feature_extractor
from ..roi_relation_predictors import make_roi_relation_predictor
from torch.nn.utils.weight_norm import weight_norm


class LinkNet(nn.Module):
    def __init__(self, cfg, in_channels):
        super(LinkNet, self).__init__()
        self.cfg = cfg
        self.in_channels = in_channels
        self.feature_extractor = make_roi_relation_feature_extractor(cfg, in_channels)
        self.predictor = make_roi_relation_predictor(cfg, self.feature_extractor.out_channels)
        C = cfg.MODEL
        L = cfg.MODEL.LINKNET
        self.K_0 = FCNet([C.ROI_BOX_HEAD.NUM_CLASSES, L.LABEL_EMBEDDING_SIZE], '', 0)
        self.K_1 = nn.Linear(C.ROI_BOX_HEAD.NUM_CLASSES, L.LABEL_EMBEDDING_SIZE, bias=False)
        self.K_2 = FCNet([L.GEOMETRIC_LAYOUT_SIZE, L.GEOMETRIC_LAYOUT_ENCODING_SIZE], '', 0)
        self.G_0 = FCNet([in_channels, C.ROI_BOX_HEAD.NUM_CLASSES], '', L.SATT_DROPOUT_RATE)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # 3.3.1 Object-Relational Embedding
        satt_input_size = self.feature_extractor.out_channels + L.LABEL_EMBEDDING_SIZE + in_channels
        satt_hid0_size = int(satt_input_size / L.SATT_HIDDEN_FACTOR)
        satt_hid1_size = int(L.OBJ_REL_EMBEDDING_SIZE / L.SATT_HIDDEN_FACTOR)
        self.obj_rel_emb = nn.ModuleList([
            SA(satt_input_size, satt_hid0_size, satt_hid0_size, glimpse=1, dropout=L.SATT_DROPOUT_RATE, ffn=False),
            FCNet([satt_input_size, L.OBJ_REL_EMBEDDING_SIZE], '', L.SATT_DROPOUT_RATE),
            SA(L.OBJ_REL_EMBEDDING_SIZE, satt_hid1_size, satt_hid1_size, glimpse=1, dropout=L.SATT_DROPOUT_RATE, ffn=False)
            ])
        self.obj_rel_classifier = FCNet([L.OBJ_REL_EMBEDDING_SIZE, C.ROI_BOX_HEAD.NUM_CLASSES], '', L.SATT_DROPOUT_RATE)

        # 3.4.1 Edge-Relational Embedding
        edge_input_size = L.OBJ_REL_EMBEDDING_SIZE + L.LABEL_EMBEDDING_SIZE
        edge_hid0_size = int(edge_input_size / L.SATT_HIDDEN_FACTOR)
        edge_hid1_size = int(L.OBJ_REL_EMBEDDING_SIZE / L.SATT_HIDDEN_FACTOR)
        self.edge_rel_emb = nn.ModuleList([
            SA(edge_input_size, edge_hid0_size, edge_hid0_size, glimpse=1, dropout=L.SATT_DROPOUT_RATE, ffn=False),
            FCNet([edge_input_size, L.OBJ_REL_EMBEDDING_SIZE], '', L.SATT_DROPOUT_RATE),
            SA(L.OBJ_REL_EMBEDDING_SIZE, edge_hid1_size, edge_hid1_size, glimpse=1, dropout=L.SATT_DROPOUT_RATE, ffn=False)
            ])
        self.edge_rel_classifier = FCNet([L.OBJ_REL_EMBEDDING_SIZE, 2 * self.feature_extractor.out_channels], '', L.SATT_DROPOUT_RATE)
        self.rel_classifier = FCNet([self.feature_extractor.out_channels + L.GEOMETRIC_LAYOUT_ENCODING_SIZE, C.ROI_RELATION_HEAD.NUM_CLASSES], '', L.SATT_DROPOUT_RATE)
        self.PADDING = 0

    def forward(self, features, proposals, proposal_pairs):
        # e.g., ResNet50Conv5ROIFeatureExtractor
        # features: list(tensor@1024xfhxfw))x1
        # proposals: list(BoxList)
        # proposal_pairs: list(BoxPairList)
        # 
        # Output:
        # x: tensor@[collapsed_pairs]x2048x7x7
        # obj_class_logits: tensor@[collapsed_boxes]x151
        # rel_class_logits: tensor@[collapsed_pairs]x51
        box_features = torch.cat([proposal.get_field("features") for proposal in proposals], 0)
        box_logits = torch.cat([proposal.get_field("logits") for proposal in proposals], 0)

        f_roi = self.avgpool(box_features).squeeze(-1).squeeze(-1)  # [collapsed_boxes]x2048
        K_0_l = self.K_0(box_logits)  # [collapsed_boxes]x200
        c, global_logits = self.global_context_embedding(features)  # bx1024, bx151
    
        padded = self.pad_packed_tensor([f_roi, K_0_l], proposals, padding_value=self.PADDING)  # bxBx2048, bxBx200
        f_roi_padded, K_0_l_padded = padded[0], padded[1]
        b_mask = f_roi_padded.abs().sum(-1)==self.PADDING

        O_0 = torch.cat([f_roi_padded, K_0_l_padded, c.unsqueeze(1).repeat(1,f_roi_padded.size(1),1)], dim=-1)  # bxBx(2048+200+1024=3272)
        O_3 = self.fwd_sa(O_0, self.obj_rel_emb, b_mask)
        O_4 = self.obj_rel_classifier(O_3)
        obj_class_logits = self.pack_padded_tensor(O_4, proposals)

        O_4a = self.one_hot_sampling(O_4)
        K_1_O_4a = self.K_1(O_4a)
        E_0 = torch.cat([K_1_O_4a, O_3], -1)  # bxBx(200+256=456)
        e_0 = self.fwd_sa(E_0, self.edge_rel_emb, b_mask)  # bxBx...
        E_1 = self.edge_rel_classifier(e_0)
        E_1_sbj = E_1[..., :self.feature_extractor.out_channels]
        E_1_obj = E_1[..., self.feature_extractor.out_channels:]

        F = self.avgpool(self.feature_extractor(features, proposal_pairs)).squeeze(-1).squeeze(-1)  # [collapsed_pairs]x...

        G_0 = self.interaction_embedding(E_1_sbj, E_1_obj, F, proposal_pairs)  # [collapsed_pairs]x...
        G_1 = torch.cat([G_0, self.K_2(self.geometric_layout_encoding(G_0, proposal_pairs))], dim=-1)
        G_2 = self.rel_classifier(G_1)
        rel_class_logits = G_2
       
        return (f_roi, F), obj_class_logits, rel_class_logits, global_logits

    @staticmethod
    def fwd_sa(x, net, mask):
        return net[2](net[1](net[0](x, mask)), mask)

    def global_context_embedding(self, features):
        assert 1==len(features)
        global_features = self.avgpool(features[0]).squeeze(-1).squeeze(-1)
        global_logits = self.G_0(global_features)
        return global_features, global_logits

    def interaction_embedding(self, sbj, obj, union, proposal_pairs):
        out = []
        idx = 0
        for i, pair in enumerate(proposal_pairs):
            idx_pairs = pair.get_field('idx_pairs')
            l = idx_pairs.size(0)
            out.append(union[idx:idx+l] * sbj[i,idx_pairs[:,0]] * obj[i,idx_pairs[:,1]])
            idx += l
        return torch.cat(out, dim=0)

    def geometric_layout_encoding(self, x, proposal_pairs, eps=1e-15):
        for i, pairs in enumerate(proposal_pairs):
            bbox_sbj = pairs.bbox[:,:4]
            bbox_obj = pairs.bbox[:,4:]
            if 'xyxy'==pairs.mode:
                layout = torch.stack([
                        (bbox_obj[:,0] - bbox_sbj[:,0]) / (bbox_sbj[:,2] - bbox_sbj[:,0] + eps),
                        (bbox_obj[:,1] - bbox_sbj[:,1]) / (bbox_sbj[:,3] - bbox_sbj[:,1] + eps),
                        ((bbox_obj[:,2] - bbox_obj[:,0]) / (bbox_sbj[:,2] - bbox_sbj[:,0] + eps) + eps).log(),
                        ((bbox_obj[:,3] - bbox_obj[:,1]) / (bbox_sbj[:,3] - bbox_sbj[:,1] + eps) + eps).log(),
                        ], dim=-1).to(x.dtype)
            else:
                raise NotImplementedError()
        return layout

    @staticmethod
    def one_hot_sampling(logits):
        inds = torch.max(logits, -1)[1]  # argmax
        one_hots = logits.new(*logits.size()).fill_(0)
        one_hots.scatter_(-1, inds.unsqueeze(-1), 1)
        return one_hots

    @staticmethod
    def pad_packed_tensor(x, proposals, padding_value=0):
        lengths, max_len = LinkNet.get_lengths(proposals)
        padded = []
        for packed in x:
            out = []
            idx = 0
            for i in range(len(proposals)):
                l = lengths[i]
                pad_dims = (max_len - l,) + packed.size()[1:]
                pad = packed.new(*pad_dims).fill_(padding_value)
                out.append(torch.cat([packed[idx:idx+l], pad], dim=0))
                idx = idx + l
            padded.append(torch.stack(out, dim=0))
        return padded

    @staticmethod
    def pack_padded_tensor(x, proposals):
        # x: tensor@bxBx...
        # out: tensor@[collapsed_boxes]x...
        lengths, max_len = LinkNet.get_lengths(proposals)
        out = []
        idx = 0
        for i in range(len(lengths)):
            l = lengths[i]
            out.append(x[i,:l])
            idx += l
        return torch.cat(out, dim=0)

    @staticmethod
    def repeat_tensor(x, proposals):
        #   x: tensor@bx1024
        # out: tensor@[collapsed_boxes]x1024
        lengths, max_len = LinkNet.get_lengths(proposals)
        trailing_dims = x.size()[1:]
        out_dims = (sum(lengths),) + trailing_dims
        out = x.new(*out_dims).fill_(0)
        idx = 0
        for i, bbox in enumerate(proposals):
            out[idx:idx+len(bbox)] = x[i].unsqueeze(0)
            idx += len(bbox)
        return out

    @staticmethod
    def get_lengths(proposals, tensorize=False):
        lengths = []
        for bbox in proposals:
            lengths.append(len(bbox))
        if not tensorize:
            return lengths, max(lengths)
        else:
            return torch.LongTensor(lengths).cuda(), max(lengths)


class GA(nn.Module):
    def __init__(self, x_dim, y_dim, h_dim, o_dim, glimpse, act='ReLU', dropout=.1, ffn=True):
        super(GA, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.h_dim = h_dim
        self.o_dim = o_dim
        self.glimpse = glimpse
        self.d_dim = math.floor(h_dim / glimpse)
        self.q_net = FCNet([x_dim, h_dim], act, dropout)
        self.optimize = True
        if self.optimize:
            self.kv_net = FCNet([y_dim, h_dim + o_dim], act, dropout)
        else:
            self.k_net = FCNet([y_dim, h_dim], act, dropout)
            self.v_net = FCNet([y_dim, o_dim], act, dropout)
        self.m_net = FCNet([o_dim, x_dim], act, dropout)
        self.ffn = ffn
        if ffn:
            self.p_net = FCNet([x_dim, 4 * x_dim, x_dim], act, dropout)
            self.p_lnz = nn.ModuleList([nn.LayerNorm(x_dim), nn.LayerNorm(x_dim)])
        else:
            self.p_lnz = nn.ModuleList([nn.LayerNorm(x_dim)])

    def forward(self, x, y, x_msk, y_msk):
        q_emb = self.q_net(x).view(x.size(0), x.size(1), self.glimpse, -1)
        if self.optimize:
            kv_emb = self.kv_net(y).view(y.size(0), y.size(1), self.glimpse, -1)
            k_emb, v_emb = kv_emb[:,:,:,:self.d_dim], kv_emb[:,:,:,self.d_dim:]
        else:
            k_emb = self.k_net(y).view(y.size(0), y.size(1), self.glimpse, -1)
            v_emb = self.v_net(y).view(y.size(0), y.size(1), self.glimpse, -1)
        logit = torch.einsum('bqgd,bkgd->bqkg', q_emb, k_emb) / torch.Tensor(1).fill_(self.d_dim).sqrt().cuda()
        logit = logit.masked_fill_(y_msk.unsqueeze(1).unsqueeze(3), -float('inf'))
        p = F.softmax(logit, 2)
        o0 = torch.einsum('bkgh,bqkg->bqgh', v_emb, p).view(x.size(0), x.size(1), -1)
        o1 = self.m_net(o0)
        o2 = self.p_lnz[0]((o1 + x))
        if self.ffn:
            o2 = self.p_lnz[1](self.p_net(o2) + o2)
        return o2


class SA(nn.Module):
    def __init__(self, x_dim, h_dim, o_dim, glimpse, act='ReLU', dropout=.1, ffn=True):
        super(SA, self).__init__()
        self.ga = GA(x_dim, x_dim, h_dim, o_dim, glimpse, act, dropout, ffn)

    def forward(self, x, x_msk):
        return self.ga(x, x, x_msk, x_msk)


class FCNet(nn.Module):
    """Simple class for non-linear fully connect network
    """
    def __init__(self, dims, act='ReLU', dropout=0, last_act=True, wn=False):
        super(FCNet, self).__init__()

        layers = []
        for i in range(len(dims)-2):
            in_dim = dims[i]
            out_dim = dims[i+1]
            if 0 < dropout:
                layers.append(nn.Dropout(dropout))
            if wn:
                layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
            else:
                layers.append(nn.Linear(in_dim, out_dim))
            if ''!=act:
                layers.append(getattr(nn, act)())
        if 0 < dropout:
            layers.append(nn.Dropout(dropout))
        if wn:
            layers.append(weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None))
        else:
            layers.append(nn.Linear(dims[-2], dims[-1]))
        if ''!=act and last_act:
            layers.append(getattr(nn, act)())

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


def build_linknet_model(cfg, in_channels):
    return LinkNet(cfg, in_channels)
