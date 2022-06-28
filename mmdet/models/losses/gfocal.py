import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import weighted_loss
import pdb
import torch
@LOSSES.register_module()
class DenisityGFocalLoss(nn.Module):
    r"""Quality Focal Loss (QFL) is a variant of `Generalized Focal Loss:
    Learning Qualified and Distributed Bounding Boxes for Dense Object
    Detection <https://arxiv.org/abs/2006.04388>`_.
    Args:
        use_sigmoid (bool): Whether sigmoid operation is conducted in QFL.
            Defaults to True.
        beta (float): The beta parameter for calculating the modulating factor.
            Defaults to 2.0.
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Loss weight of current loss.
    """

    def __init__(self,
                 beta=2.0,
                 reduction = 'mean',
                 loss_weight=1.0):
        super(DenisityGFocalLoss, self).__init__()
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,preds, targets):
        ''' Modified focal loss. Exactly the same as CornerNet.
            Runs faster and costs a little bit more memory
            Arguments:
            preds (B x c x h x w)
            gt_regr (B x c x h x w)
        '''
        #pdb.set_trace()
        pos_inds = targets.gt(0).float()# heatmap为1的部分是正样本
        neg_inds = targets.lt(0).float()# 其他部分为负样本

        loss = 0
        preds = torch.clamp(preds, min=1e-4, max=1 - 1e-4)
        '''
        for i,pred in enumerate(preds): # 预测值
            # 约束在0-1之间
            pred = torch.clamp(pred, min=1e-4, max=1 - 1e-4)
            pos_loss = (targets[i]*torch.log(pred) + (1-targets[i])*torch.log(1-pred)) * torch.pow((targets[i] - pred).abs(), self.beta) * pos_inds[i]
            neg_loss = torch.log(1 - pred) * torch.pow(pred,self.beta)  * neg_inds[i]
            pos_loss = pos_loss.sum()
            neg_loss = neg_loss.sum()
            loss = loss - (pos_loss + neg_loss)
        '''
        loss = -torch.pow((targets - preds).abs(), self.beta) * ( targets*torch.log(preds) + (1-targets)*torch.log(1-preds) )
        if self.reduction == 'mean':
            return self.loss_weight * loss.mean()
        elif self.reduction == 'sum':
            return self.loss_weight * loss.sum()
        elif self.reduction == 'none':
            return self.loss_weight * loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))