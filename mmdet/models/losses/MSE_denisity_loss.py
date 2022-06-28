import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import weighted_loss
import pdb
import torch
import numpy as np

import cv2
import os
def save_density_map(density_map, output_dir = 'denisity.jpg'):

    density_map = 255.0 * (density_map - np.min(density_map) + 1e-10) / (1e-10 + np.max(density_map) - np.min(density_map))
    density_map = density_map.squeeze()
    color_map = cv2.applyColorMap(density_map[:, :, np.newaxis].astype(np.uint8).repeat(3, axis=2), cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join(output_dir), color_map)
@LOSSES.register_module()
class DenisityWMSELoss(nn.Module):
    def __init__(self,loss_weight=1.0):
        super(DenisityWMSELoss, self).__init__()
        self.loss_weight = loss_weight
        self.enhance_feature_loss = nn.MSELoss(reduce=False, size_average=False)
    def forward(self,preds, targets):
        loss = 0
        #save_density_map(targets[0][0].detach().cpu().numpy())
        for j in range(preds.size(0)):
            positive_weight_den = (torch.gt(targets[j],0)*1).sum().float()/(targets[j].shape[1]*targets[j].shape[2])
            weight_matrix_den = torch.where( torch.gt(targets[j],0) ,torch.full_like(targets[0], 1-positive_weight_den),torch.full_like(targets[0],positive_weight_den ))
            loss = loss + (self.enhance_feature_loss(preds[j],targets[j]) * weight_matrix_den).sum()/(torch.gt(targets[j],0)*1).sum().float()
            if torch.isnan(loss):
            	import pdb;pdb.set_trace()
        #import pdb;pdb.set_trace()
        #print(loss)
        return loss * self.loss_weight