import torch
import torch.nn as nn
import torch.nn.functional as F

'''
Modules for making AdvMCD (Xu et al. 2019)

'''

class CFMLayer(nn.Module):
    def __init__(self, ch=64):
        super(CFMLayer, self).__init__()
        self.CFM_scale_conv0 = nn.Conv2d(ch//2, ch//2, 1)
        self.CFM_scale_conv1 = nn.Conv2d(ch//2, ch, 1)
        self.CFM_shift_conv0 = nn.Conv2d(ch//2, ch//2, 1)
        self.CFM_shift_conv1 = nn.Conv2d(ch//2, ch, 1)

    def forward(self, x):
        # # x[0]: fea; x[1]: cond
        scale = self.CFM_scale_conv1(F.leaky_relu(self.CFM_scale_conv0(x[1]), 0.1, inplace=True))
        shift = self.CFM_shift_conv1(F.leaky_relu(self.CFM_shift_conv0(x[1]), 0.1, inplace=True))
        return x[0] * (scale + 1) + shift


class ResBlock_CFM(nn.Module):
    def __init__(self, ch=64):
        super(ResBlock_CFM, self).__init__()
        self.CFM0 = CFMLayer(ch)
        self.conv0 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.CFM1 = CFMLayer(ch)
        self.conv1 = nn.Conv2d(ch, ch, 3, 1, 1)

    def forward(self, x):
        # x[0]: fea; x[1]: cond
        fea = self.CFM0(x)
        fea = F.relu(self.conv0(fea), inplace=True)
        fea = self.CFM1((fea, x[1]))
        fea = self.conv1(fea)
        return (x[0] + fea, x[1])