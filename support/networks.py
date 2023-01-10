import torch
import torch.nn as nn
import torch.nn.functional as F

from .WCMC import ConvChain

'''
Ensembling networks for our work
'''

class InterpolationNet(nn.Module):
    def __init__(self, n_in, depth=9, width=50, activation='softmax', model_type='conv5'):
        super(InterpolationNet, self).__init__()

        self.activation = activation

        self.model = ConvChain(
            n_in, 2, depth=depth, width=width, ksize=5,
            activation="relu", weight_norm=False, pad=True,
            output_type='linear')

    def forward(self, data):
        if self.activation == 'softmax':
            return F.softmax(self.model(data), dim=1)
        else:
            return torch.sigmoid(self.model(data))
        