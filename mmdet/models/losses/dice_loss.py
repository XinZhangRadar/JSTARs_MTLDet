import torch
import torch.nn as nn
from ..builder import LOSSES
import cv2
@LOSSES.register_module()
class DiceLoss(nn.Module):
    def __init__(self, branch = None, smooth=1, p=2, reduction='mean',loss_weight=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.branch = branch
        if self.loss_weight == 'auto' or self.loss_weight == 'auto2' :
            params = torch.ones(1, requires_grad=True)
            self.Auto_loss_weight = torch.nn.Parameter(params)



    def forward(self, predict, target,labels = None):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        
        #cv2.imwrite('edge.jpg',target[0][0].detach().cpu().numpy()*255)
        #import pdb;pdb.set_trace()

        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)


        num = torch.sum(torch.mul(predict, target), dim=1) * 2 + self.smooth
        den = torch.sum((predict.pow(self.p) + target.pow(self.p)), dim=1) + self.smooth

        loss = 1 - num / den
        #import pdb;pdb.set_trace()
        if self.loss_weight == 'auto':
            #print(self.branch)
            #print(0.5 / (self.Auto_loss_weight ** 2))
            return 0.5 / (self.Auto_loss_weight ** 2) * loss + torch.log(self.Auto_loss_weight)
        elif self.loss_weight == 'auto2':
            #print(self.branch)
            #print(0.5 / (self.Auto_loss_weight ** 2))
            return 0.5 / (self.Auto_loss_weight ** 2) * loss + torch.log(1+ self.Auto_loss_weight ** 2)
        else:
            if self.reduction == 'mean':
                return self.loss_weight * loss.mean()
            elif self.reduction == 'sum':
                return self.loss_weight * loss.sum()
            elif self.reduction == 'none':
                return self.loss_weight * loss
            else:
                raise Exception('Unexpected reduction {}'.format(self.reduction))