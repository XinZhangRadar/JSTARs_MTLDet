import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, kaiming_init
import torch
from mmdet.core import auto_fp16, force_fp32
from mmdet.models.builder import HEADS
import numpy as np
import cv2
import os

@HEADS.register_module()
class FusedDenisityHead(nn.Module):
    r"""Multi-level fused denisity head.

    .. code-block:: none

        in_1 -> 1x1 conv ---
                            |
        in_2 -> 1x1 conv -- |
                           ||
        in_3 -> 1x1 conv - ||
                          |||                  /-> 1x1 conv (denisity prediction)
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
                 num_classes=1,
                 ignore_label=255,
                 loss_weight=0.2,
                 conv_cfg=None,
                 norm_cfg=None):
        super(FusedDenisityHead, self).__init__()
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

        self.criterion = nn.MSELoss(reduce=False, size_average=False)

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
        reconstruct_size = (fused_size[0]*stride,fused_size[1]*stride)
        #import pdb;pdb.set_trace()

        #denisity_pred = F.interpolate(self.conv_logits(x), size=reconstruct_size, mode='bilinear', align_corners=True)
        denisity_pred = self.conv_logits(x)
        denisity_pred = torch.sigmoid(denisity_pred)
        #save_density_map(denisity_pred[0][0].detach().cpu().numpy())

        #denisity_pred_attention = torch.sigmoid(self.conv_logits(x))
        #denisity_pred_attention_mask = denisity_pred_attention>0
        #denisity_pred_attention = denisity_pred_attention * denisity_pred_attention_mask


        x = self.conv_embedding(x)# * denisity_pred_attention
        return denisity_pred, x

def save_density_map(density_map, output_dir = 'denisity.jpg'):

    density_map = 255.0 * (density_map - np.min(density_map) + 1e-10) / (1e-10 + np.max(density_map) - np.min(density_map))
    density_map = density_map.squeeze()
    color_map = cv2.applyColorMap(density_map[:, :, np.newaxis].astype(np.uint8).repeat(3, axis=2), cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join(output_dir), color_map)