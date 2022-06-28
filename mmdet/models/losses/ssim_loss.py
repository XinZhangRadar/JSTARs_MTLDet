import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import functional as F

import numpy as np
from ..builder import LOSSES
from .utils import weighted_loss

@weighted_loss
def gaussian_kernel(size, sigma):
    x, y = np.mgrid[-size:size+1, -size:size+1]
    kernel = np.exp(-0.5*(x*x+y*y)/(sigma*sigma))
    kernel /= kernel.sum()
    return kernel


@LOSSES.register_module()
class SSIM_Loss(nn.Module):
    def __init__(self, in_channels, size=11, sigma=1.5, size_average=True):
        super(SSIM_Loss, self).__init__()
        self.in_channels = in_channels
        self.size = int(size)
        self.sigma = sigma
        self.size_average = size_average

        kernel = gaussian_kernel(self.size, self.sigma)
        self.kernel_size = kernel.shape
        weight = np.tile(kernel, (in_channels, 1, 1, 1))
        self.weight = Parameter(torch.from_numpy(weight).float(), requires_grad=False)

    def forward(self, input, target, mask=None):
        import pdb;pdb.set_trace()
        mean1 = F.conv2d(input, self.weight, padding=self.size, groups=self.in_channels)
        mean2 = F.conv2d(target, self.weight, padding=self.size, groups=self.in_channels)
        mean1_sq = mean1*mean1
        mean2_sq = mean2*mean2
        mean_12 = mean1*mean2

        sigma1_sq = F.conv2d(input*input, self.weight, padding=self.size, groups=self.in_channels) - mean1_sq
        sigma2_sq = F.conv2d(target*target, self.weight, padding=self.size, groups=self.in_channels) - mean2_sq
        sigma_12 = F.conv2d(input*target, self.weight, padding=self.size, groups=self.in_channels) - mean_12
    
        C1 = 0.01**2
        C2 = 0.03**2

        ssim = ((2*mean_12+C1)*(2*sigma_12+C2)) / ((mean1_sq+mean2_sq+C1)*(sigma1_sq+sigma2_sq+C2))
        if self.size_average:
            out = 1 - ssim.mean()
        else:
            out = 1 - ssim.view(ssim.size(0), -1).mean(1)
        return out