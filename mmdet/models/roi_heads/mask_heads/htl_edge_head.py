from mmcv.cnn import ConvModule

from mmdet.models.builder import HEADS,build_loss
from .fcn_mask_head import FCNMaskHead
import torch

@HEADS.register_module()
class HTLEdgeHead(FCNMaskHead):

    def __init__(self, with_conv_res=True, loss_edge = None,*args, **kwargs):
        super(HTLEdgeHead, self).__init__(*args, **kwargs)
        self.with_conv_res = with_conv_res
        if self.with_conv_res:
            self.conv_res = ConvModule(
                self.conv_out_channels,
                self.conv_out_channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg)
        self.loss_mask = build_loss(loss_edge)
    def init_weights(self):
        super(HTLEdgeHead, self).init_weights()
        if self.with_conv_res:
            self.conv_res.init_weights()

    def forward(self, x):
        x = self.upsample(x)
        if self.upsample_method == 'deconv':
            x = self.relu(x)
        edge_mask_pred = torch.sigmoid(self.conv_logits(x))
        return edge_mask_pred 
