import torch
import torch.nn as nn
from ..builder import LOSSES
import cv2
@LOSSES.register_module()
class WCE_Loss(nn.Module):
    def __init__(self, w_positive = 1, w_negative = 1.1):
        super(WCE_Loss, self).__init__()
        self.w_positive = w_positive
        self.w_negative = w_negative 


    def forward(self,prediction, label):
       
        label = label.long()
        mask = label.float()
        num_positive = torch.sum((mask==1).float()).float()
        num_negative = torch.sum((mask==0).float()).float()
        cv2.imwrite('edge.jpg',label[0][0].detach().cpu().numpy()*255)
        mask[mask == 1] = 1.0 * num_negative / (num_positive + num_negative)
        mask[mask == 0] = 1.1 * num_positive / (num_positive + num_negative)
        cost = torch.nn.functional.binary_cross_entropy(
                prediction.float(),label.float(), weight=mask, reduce=False)
        import pdb;pdb.set_trace()
        return cost#cost/num_positive#torch.mean(cost)



        #cost = torch.nn.functional.binary_cross_entropy( prediction.float(),label.float(),reduce=False)