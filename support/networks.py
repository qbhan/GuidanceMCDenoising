from turtle import forward
from pyparsing import Forward
import torch
import torch.nn as nn
import sys
import configs
sys.path.insert(1, configs.PATH_SBMC)
try:
    from sbmc import modules as ops
except ImportError as error:
    print('Put appropriate paths in the configs.py file.')
    raise

# imports for second stragety
import numpy as np
import torch.nn.functional as F
from support.utils import crop_like
from support.modules import SimpleUNet, Generator, Discriminator

'''
Ensembling networks for our work
'''

class InterpolationNet(nn.Module):
    def __init__(self, n_in, depth=9, width=50, activation='softmax', model_type='conv5'):
        super(InterpolationNet, self).__init__()

        self.activation = activation

        if model_type == 'conv5':
            self.model = ops.ConvChain(
                n_in, 2, depth=depth, width=width, ksize=5,
                activation="relu", weight_norm=False, pad=True,
                output_type='linear')
        elif model_type == 'unet':
            self.model = SimpleUNet(n_in, 2, activation='leaky_relu')

    def forward(self, data):
        if self.activation == 'softmax':
            return F.softmax(self.model(data), dim=1)
        else:
            return torch.sigmoid(self.model(data))

class InterpolationNet2(nn.Module):
    def __init__(self, n_in, depth=9, width=100, activation='softmax', model_type='conv5'):
        super(InterpolationNet2, self).__init__()

        self.activation = activation

        self.model = ops.ConvChain(
                75, 2, depth=depth, width=width, ksize=5,
                activation="relu", weight_norm=False, pad=True,
                output_type='linear')

        self.embed_rad = nn.Sequential(
            nn.Conv2d(6, 25, kernel_size=1, padding=0),
            nn.ReLU(),
        )
        self.embed_Gbuf = nn.Sequential(
            nn.Conv2d(24, 25, kernel_size=1, padding=0),
            nn.ReLU(),    
        )
        self.embed_Pbuf = nn.Sequential(
            nn.Conv2d(13, 25, kernel_size=1, padding=0),
            nn.ReLU()
        )

    def forward(self, data):
        if self.activation == 'softmax':
            # print(data.shape)
            x = torch.cat([self.embed_rad(data[:, :6]), self.embed_Gbuf(data[:, 6:30]), self.embed_Pbuf(data[:, 30:])], dim=1)
            # print(x.shape)
            return F.softmax(self.model(x), dim=1)
        else:
            return torch.sigmoid(self.model(data))


class AdvMCD_Generator(nn.Module):
    def __init__(self):
        super(AdvMCD_Generator, self).__init__()
        
        self.diffuse = Generator()
        self.specular = Generator()

    def forward(self, data):
        # parse inputs for AdvMCD
        r_diff = data['kpcn_diffuse_buffer']
        f_diff = torch.cat((data['kpcn_diffuse_in'][:, 10:13], data['kpcn_diffuse_in'][:, 20:21], data['kpcn_diffuse_in'][:, 24:27]))
        r_spec = data['kpcn_specular_buffer']
        f_spec = torch.cat((data['kpcn_specular_in'][:, 10:13], data['kpcn_specular_in'][:, 20:21], data['kpcn_specular_in'][:, 24:27]))

        # denoise
        d_diff = self.diffuse((r_diff, f_diff))
        d_spec = self.specular((r_spec, f_spec))

        # final radiance via albedo multiplication & reverse log transformation
        albedo = crop_like(data['kpcn_albedo'], d_diff)
        d_final = (albedo * d_diff) + (torch.exp(d_spec) - 1.0)

        output = dict(radiance=d_final, diffuse=d_diff, specular=d_spec)
        return output
        

class AdvMCD_Discriminator(nn.Module):
    def __init__(self):
        super(AdvMCD_Discriminator, self).__init__()

        self.diffuse = Discriminator()
        self.specular = Discriminator()

    def forward(self, data):
        '''
        fake['diffuse'], fake['specular']
        real['diffuse'], real['specular']
        '''
        output = dict(diffuse=self.diffuse(data['diffuse']), specular=self.specular(data['specular']))
        return output

        

class PathNet(nn.Module):
    """Path embedding network (Cho et al. 2021)
    """

    def __init__(self, ic, intermc=64, outc=3):
        super(PathNet, self).__init__()
        self.ic = ic
        self.intermc = intermc
        self.outc = outc
        self.final_ic = intermc + intermc

        self.embedding = ops.ConvChain(ic, intermc, width=intermc, depth=3,
                ksize=1, pad=False)
        self.propagation = ops.Autoencoder(intermc, intermc, num_levels=3, 
                increase_factor=2.0, num_convs=3, width=intermc, ksize=3,
                output_type="leaky_relu", pooling="max")
        self.final = ops.ConvChain(self.final_ic, outc, width=self.final_ic, 
                depth=2, ksize=1, pad=False, output_type="relu")

    def __str__(self):
        return "PathNet i{}in{}o{}".format(self.ic, self.intermc, self.outc)

    def forward(self, samples):
        paths = samples["paths"]
        bs, spp, nf, h, w = paths.shape

        flat = paths.contiguous().view([bs*spp, nf, h, w])
        flat = self.embedding(flat)
        flat = flat.view([bs, spp, self.intermc, h, w])
        reduced = flat.mean(1)

        propagated = self.propagation(reduced)
        flat = torch.cat([flat.view([bs*spp, self.intermc, h, w]), propagated.unsqueeze(1).repeat(
            [1, spp, 1, 1, 1]).view(bs*spp, self.intermc, h, w)], 1)
        out = self.final(flat).view([bs, spp, self.outc, h, w])
        return out