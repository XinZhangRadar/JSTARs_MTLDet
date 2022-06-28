import torch.nn as nn
import torch

from ..builder import LOSSES
from .utils import weighted_loss
import numpy as np

import cv2
import os

@weighted_loss
def denisity_focal_loss(pred, gaussian_target, alpha=2.0, gamma=4.0):
    """`Focal Loss <https://arxiv.org/abs/1708.02002>`_ for targets in gaussian
    distribution.

    Args:
        pred (torch.Tensor): The prediction.
        gaussian_target (torch.Tensor): The learning target of the prediction
            in gaussian distribution.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 2.0.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 4.0.
    """
    #import pdb;pdb.set_trace()
    eps = 1e-12
    max_v = gaussian_target.max()
    pos_weights = gaussian_target.eq(max_v)
    neg_weights = (1 - gaussian_target).pow(gamma)
    pos_loss = -(pred + eps).log() * (1 - pred).pow(alpha)*pos_weights
    neg_loss = -(1 - pred + eps).log() * pred.pow(alpha) *neg_weights
    #save_density_map(gaussian_target[0][0].detach().cpu().numpy())

    return pos_loss + neg_loss

def save_density_map(density_map, output_dir = 'denisity.jpg'):

    density_map = 255.0 * (density_map - np.min(density_map) + 1e-10) / (1e-10 + np.max(density_map) - np.min(density_map))
    density_map = density_map.squeeze()
    color_map = cv2.applyColorMap(density_map[:, :, np.newaxis].astype(np.uint8).repeat(3, axis=2), cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join(output_dir), color_map)

@LOSSES.register_module()

class DenisityFocalLoss(nn.Module):
    """DenisityFocalLoss is a variant of focal loss.

    More details can be found in the `paper
    <https://arxiv.org/abs/1808.01244>`_
    Code is modified from `kp_utils.py
    <https://github.com/princeton-vl/CornerNet/blob/master/models/py_utils/kp_utils.py#L152>`_  # noqa: E501
    Please notice that the target in GaussianFocalLoss is a gaussian heatmap,
    not 0/1 binary target.

    Args:
        alpha (float): Power of prediction.
        gamma (float): Power of target for negtive samples.
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Loss weight of current loss.
    """

    def __init__(self,
                 alpha=2.0,
                 gamma=4.0,
                 reduction='mean',
                 loss_weight=1.0):
        super(DenisityFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction
                in gaussian distribution.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_reg =  denisity_focal_loss(
            pred,
            target,
            weight,
            alpha=self.alpha,
            gamma=self.gamma,
            reduction=reduction,
            avg_factor=avg_factor)
        return loss_reg * self.loss_weight


'''
@LOSSES.register_module()
class DenisityFocalLoss(nn.Module):
    def __init__(self,
                 alpha=2.0,
                 gamma=4.0,
                 reduction='mean',
                 loss_weight=1.0):
        super(DenisityFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.loss_weight = loss_weight



    def forward(self,preds, targets):
  
        import pdb;pdb.set_trace()
        pos_inds = targets.gt(0.5).float()# heatmap为1的部分是正样本
        neg_inds = targets.lt(0.5).float()# 其他部分为负样本

        neg_weights = torch.pow(1 - targets, self.gamma)# 对应(1-Yxyc)^4

        loss = 0
        for i, pred in enumerate(preds): # 预测值
            # 约束在0-1之间
            pred = torch.clamp(pred, min=1e-4, max=1 - 1e-4)
            pos_loss = torch.log(pred) * torch.pow(1 - pred, self.alpha) * pos_inds[i]
            neg_loss = torch.log(1 - pred) * torch.pow(pred,
                                                       self.alpha) * neg_weights * neg_inds[i]
            num_pos = pos_inds.float().sum()
            pos_loss = pos_loss.sum()
            neg_loss = neg_loss.sum()

            if num_pos == 0:
                loss = loss - neg_loss # 只有负样本
            else:
                loss = loss - (pos_loss + neg_loss) / num_pos
        return loss / len(preds)
'''