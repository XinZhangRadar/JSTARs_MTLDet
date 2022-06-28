import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, kaiming_init

from mmdet.core import auto_fp16, force_fp32
from mmdet.models.builder import HEADS
import numpy as np
import torch
import cv2
@HEADS.register_module()
class FusedEdgeHead(nn.Module):
    r"""Multi-level fused Edge head.

    .. code-block:: none

        in_1 -> 1x1 conv ---
                            |
        in_2 -> 1x1 conv -- |
                           ||
        in_3 -> 1x1 conv - ||
                          |||                  /-> 1x1 conv (edge prediction)
        in_4 -> 1x1 conv -----> 3x3 convs (*4)
                            |                  \-> 1x1 conv (feature)
        in_5 -> 1x1 conv ---
    """  # noqa: W605

    def __init__(self,
                 num_ins,
                 fusion_level,
                 num_convs=4,
                 in_channels=256,
                 conv_out_channels=256,
                 num_classes=183,
                 ignore_label=255,
                 loss_weight=0.2,
                 conv_cfg=None,
                 norm_cfg=None):
        super(FusedEdgeHead, self).__init__()
        self.num_ins = num_ins
        self.fusion_level = fusion_level
        self.num_convs = num_convs
        self.in_channels = in_channels
        self.conv_out_channels = conv_out_channels
        self.num_classes = num_classes
        self.ignore_label = ignore_label
        self.loss_weight = loss_weight
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.fp16_enabled = False


        self.lateral_convs = nn.ModuleList()
        for i in range(self.num_ins):
            self.lateral_convs.append(
                ConvModule(
                    self.in_channels,
                    self.in_channels,
                    1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    inplace=False))

        self.convs = nn.ModuleList()
        for i in range(self.num_convs):
            in_channels = self.in_channels if i == 0 else conv_out_channels
            self.convs.append(
                ConvModule(
                    in_channels,
                    conv_out_channels,
                    3,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        self.conv_embedding = ConvModule(
            conv_out_channels,
            conv_out_channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg)
        self.conv_logits = nn.Conv2d(conv_out_channels, self.num_classes, 1)

        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_label)

    def init_weights(self):
        kaiming_init(self.conv_logits)

    @auto_fp16()
    def forward(self, feats):
        #import pdb;pdb.set_trace()
        x = self.lateral_convs[self.fusion_level](feats[self.fusion_level])
        fused_size = tuple(x.shape[-2:])
        for i, feat in enumerate(feats):
            if i != self.fusion_level:
                feat = F.interpolate(
                    feat, size=fused_size, mode='bilinear', align_corners=True)
                x += self.lateral_convs[i](feat)

        for i in range(self.num_convs):
            x = self.convs[i](x)
        stride = 2**(self.num_ins-(self.fusion_level+1))
        #import pdb,cv2
        #import pdb;pdb.set_trace()
        reconstruct_size = (fused_size[0]*stride,fused_size[1]*stride)
        edge_pred = (F.interpolate(self.conv_logits(x), size=reconstruct_size, mode='bilinear', align_corners=True))
        #edge_pred = self.conv_logits(x)
        edge_pred = torch.sigmoid(edge_pred)
        #edge_pred_attention = torch.sigmoid(self.conv_logits(x))
        #edge_pred_attention_mask = edge_pred_attention>0.9
        #edge_pred_attention = edge_pred_attention * edge_pred_attention_mask
        x = self.conv_embedding(x)# * edge_pred_attention
        #cv2.imwrite('edge.jpg',edge_pred[0][0].detach().cpu().numpy()*255)
        return edge_pred, x
