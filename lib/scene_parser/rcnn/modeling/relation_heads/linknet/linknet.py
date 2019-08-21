# LinkNet model (Woo et al., 2018)
# Reimplemented by Jin-Hwa Kim (jnhwkim@sktbrain.com)
import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence
import torch.nn.functional as F
from ..roi_relation_box_predictors import make_roi_relation_box_predictor
from ..roi_relation_feature_extractors import make_roi_relation_feature_extractor
from ..roi_relation_predictors import make_roi_relation_predictor

class LinkNet(nn.Module):
    def __init__(self, cfg, in_channels):
        super(LinkNet, self).__init__()
        self.cfg = cfg
        self.in_channels = in_channels
        self.feature_extractor = make_roi_relation_feature_extractor(cfg, in_channels)
        self.predictor = make_roi_relation_predictor(cfg, self.feature_extractor.out_channels)
        self.box_predictor = make_roi_relation_box_predictor(cfg, self.feature_extractor.out_channels)

        self.K_0 = nn.Linear(cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES, cfg.MODEL.LINKNET.LABEL_EMBEDDING_SIZE, bias=False)
        self.G_0 = nn.Linear(in_channels, cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES)

        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, features, proposals, proposal_pairs):
        # e.g., ResNet50Conv5ROIFeatureExtractor
        # features: list(tensor@bx1024xfhxfw))
        # proposals: list(BoxList)
        # proposal_pairs: list(BoxPairList)
        # 
        # Output:
        # x: tensor@[collapsed_pairs]x2048x7x7
        # obj_class_logits: tensor@[collapsed_boxes]x151
        # rel_class_logits: tensor@[collapsed_pairs]x51
        
        box_features = self.feature_extractor.get_box_feature(features, proposals)  # [collapsed_boxes]x2048x7x7
        f_roi = self.avgpool(box_features)  # [collapsed_boxes]x2048
        l = self.sampling(self.box_predictor(f_roi, averaged=True))  # [collapsed_boxes]x51
        K_0l = self.K_0(l)  # [collapsed_boxes]x200
    
        f_roi, K_0l = self.pad_packed_tensor([f_roi, K_0l], proposals)
        c, global_logits = global_context_embedding(features)  # bx1024

        o = torch.cat([f_roi, K_0l, c], dim=-1) # [collapsed_boxes]x(2048+200+1024=3272)


        x = self.feature_extractor(features, proposal_pairs)
        rel_class_logits = self.predictor(x)
       
        return (f_roi, x), obj_class_logits, rel_class_logits, global_logits

    def global_context_embedding(self, features):
        global_features = self.avgpool(features)
        global_logits = self.G_0()
        return global_features, global_logits

    def sampling(self, logits):
        inds = torch.max(logits, -1)[1]  # argmax
        one_hots = torch.zeros(*logits.size()).cuda()
        one_hots.scatter_(1, inds.view(-1, 1), 1)
        return one_hots

    @staticmethod
    def pad_packed_tensor(x, proposals):
        lengths = self.get_lengths(proposals, tensorize=True)
        padded = []
        for packed in x:
            ps = PackedSequence(packed, lengths)
            padded.append(pad_packed_sequence(ps, batch_first=True, padding_value=0))
        return *padded

    @staticmethod
    def get_lengths(proposals, tensorize=False):
        lengths = []
        for bbox in proposals:
            lengths.append(len(bbox))
        if not tensorize:
            return lengths
        else:
            return torch.Tensor(lengths).cuda()

def build_linknet_model(cfg, in_channels):
    return LinkNet(cfg, in_channels)
