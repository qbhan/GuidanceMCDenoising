import torch
import torch.nn as nn
import sys
import configs
sys.path.insert(1, configs.PATH_SBMC)
try:
    from sbmc import modules as ops
    from sbmc import functions as funcs
except ImportError as error:
    print('Put appropriate paths in the configs.py file.')
    raise
# from sbmc import modules as ops

# imports for second stragety
import numpy as np
import torch.nn.functional as F
from support.utils import crop_like
from support.modules import SimpleUNet, ModifiedConvChain
# from torch_utils import WeightedFilter

class PathNet(nn.Module):
    """Path embedding network
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


class AdvKPCN(nn.Module):
    """Re-implementation of [Bako 2017].

    Kernel-Predicting Convolutional Networks for Denoising Monte Carlo
    Renderings: <http://cvc.ucsb.edu/graphics/Papers/SIGGRAPH2017_KPCN/>

    Args:
        n_in(int): number of input channels in the diffuse/specular streams.
        ksize(int): size of the gather reconstruction kernel.
        depth(int): number of conv layers in each branch.
        width(int): number of feature channels in each branch.
    """

    def __init__(self, n_in, feat_in=50, ksize=21, depth=9, width=100, pnet_out=5):
        super(AdvKPCN, self).__init__()

        self.ksize = ksize
        self.pnet_size = pnet_out + 2
        self.feat_in = feat_in
        gbuf_in = n_in - pnet_out - 2

        self.p_diffuse = ops.ConvChain(
            self.feat_in, ksize*ksize, depth=depth, width=width, ksize=3,
            activation="relu", weight_norm=False, pad=True,
            output_type="linear")

        self.p_specular = ops.ConvChain(
            self.feat_in, ksize*ksize, depth=depth, width=width, ksize=3,
            activation="relu", weight_norm=False, pad=True,
            output_type="linear")

        self.d_diffuse = PixelDiscriminator(n_in-7, feat_in+1)
        self.d_specular = PixelDiscriminator(n_in-7, feat_in+1)
        # self.d_diffuse = PixelDiscriminator(41, feat_in+1)
        # self.d_specular = PixelDiscriminator(41, feat_in+1)

        self.g_diffuse = ops.ConvChain(
            gbuf_in, ksize*ksize, depth=depth, width=width, ksize=5,
            activation="relu", weight_norm=False, pad=False,
            output_type="linear")

        self.g_specular = ops.ConvChain(
            gbuf_in, ksize*ksize, depth=depth, width=width, ksize=5,
            activation="relu", weight_norm=False, pad=False,
            output_type="linear")

        # self.softmax = nn.Softmax(dim=1)
        # self.softmax = nn.Identity()
        self.kernel_apply = ops.KernelApply(softmax=True, splat=False)
        # self.kernel_apply = WeightedFilter(channels=3, kernel_size=ksize, bias=False, splat=False)

    def load_pretrain(self, pth):
        state = torch.load(pth)
        own_state = self.state_dict()
        # load KPCN to gbuf
        for k, v in state.items():
            if k == 'state_dict_dncnn':
                for key, value in v.items():
                    # key, value = v
                    new_key = 'gbuf_'+ key # add 'gbuf' in front of keys
                    if k not in own_state:
                        continue
                    own_state[new_key].copy_(value.data)

        # freeze these parts
        for param in self.gbuf_diffuse.parameters():
            param.requires_grad = False
        for param in self.gbuf_specular.parameters():
            param.requires_grad = False


    def forward(self, data, vis=False):
        """Forward pass of the model.

        Args:
            data(dict) with keys:
                "kpcn_diffuse_in":
                "kpcn_specular_in":
                "kpcn_diffuse_buffer":
                "kpcn_specular_buffer":
                "kpcn_albedo":

        Returns:
            (dict) with keys:
                "radiance":
                "diffuse":
                "specular":
        """
        # Process the diffuse and specular channels independently
        k_gbuf_diffuse = self.g_diffuse(data["kpcn_diffuse_in"][:, :-self.pnet_size])
        k_gbuf_specular = self.g_specular(data["kpcn_specular_in"][:,:-self.pnet_size])

        b_diffuse = crop_like(data["kpcn_diffuse_buffer"],
                              k_gbuf_diffuse).contiguous()
        b_specular = crop_like(data["kpcn_specular_buffer"],
                               k_gbuf_specular).contiguous()

        gbuf_r_diffuse = self.kernel_apply(b_diffuse, k_gbuf_diffuse)
        gbuf_r_specular = self.kernel_apply(b_specular, k_gbuf_specular)

        albedo = crop_like(data["kpcn_albedo"], gbuf_r_diffuse).contiguous()
        gbuf_final_specular = torch.exp(gbuf_r_specular) - 1
        gbuf_final_diffuse = albedo * gbuf_r_diffuse
        gbuf_final_radiance = gbuf_final_diffuse + gbuf_final_specular


        # discriminator
        batch = {
            'target_diffuse': torch.cat([data['target_diffuse'], data["kpcn_diffuse_in"][:,10:]], 1),
            'target_specular': torch.cat([data['target_specular'], data["kpcn_specular_in"][:,10:]], 1),
            'kpcn_diffuse_in': torch.cat([gbuf_r_diffuse, crop_like(data["kpcn_diffuse_in"][:,10:], gbuf_r_diffuse)], 1),
            'kpcn_specular_in': torch.cat([gbuf_r_specular, crop_like(data["kpcn_specular_in"][:,10:], gbuf_r_specular)], 1),
        }
        # dis_diffuse = self.d_diffuse(torch.cat([gbuf_final_diffuse, data["kpcn_diffuse_in"][:,10:]], 1))
        # dis_specular = self.d_specular(torch.cat([gbuf_final_specular, data["kpcn_specular_in"][:,10:]], 1))
        diff_out = self.d_diffuse(batch, mode='diff')
        spec_out = self.d_specular(batch, mode='spec')
        fake_diffuse, real_diffuse, feat_diffuse = diff_out['fake'], diff_out['real'], diff_out['feat']
        fake_specular, real_specular, feat_specular = spec_out['fake'], spec_out['real'], spec_out['feat']

        # Match dimensions
        k_diffuse = self.p_diffuse(feat_diffuse)
        k_specular = self.p_specular(feat_specular)
        # k_diffuse = self.p_diffuse(torch.cat((fake_diffuse, feat_diffuse), dim=1))
        # k_specular = self.p_specular(torch.cat((fake_specular, feat_specular), dim=1))
        b_diffuse = crop_like(data["kpcn_diffuse_buffer"],
                              k_diffuse).contiguous()
        b_specular = crop_like(data["kpcn_specular_buffer"],
                               k_specular).contiguous()
        
        # Skip Connection
        k_diffuse += k_gbuf_diffuse * fake_diffuse
        k_specular += k_gbuf_specular * fake_specular
        
        r_diffuse = self.kernel_apply(b_diffuse, self.softmax(k_diffuse))
        r_specular = self.kernel_apply(b_specular, self.softmax(k_specular))

        albedo = crop_like(data["kpcn_albedo"], r_diffuse).contiguous()
        final_specular = torch.exp(r_specular) - 1
        final_diffuse = albedo * r_diffuse
        final_radiance = final_diffuse + final_specular

        if not vis:
            output = dict(radiance=final_radiance, diffuse=r_diffuse,
                    specular=r_specular, g_radiance=gbuf_final_radiance,
                    g_diffuse=gbuf_r_diffuse, g_specular=gbuf_r_specular,
                    s_f_diffuse=fake_diffuse, s_f_specular=fake_specular,
                    s_r_diffuse=real_diffuse, s_r_specular=real_specular,
                    k_diffuse=k_diffuse, k_specular=k_specular,
                    k_g_diffuse=k_gbuf_diffuse, k_g_specular=k_gbuf_specular)

        else:
            output = dict(radiance=final_radiance, diffuse=r_diffuse,
                    specular=r_specular, g_radiance=gbuf_final_radiance,
                    g_diffuse=gbuf_r_diffuse, g_specular=gbuf_r_specular,
                    s_f_diffuse=fake_diffuse, s_f_specular=fake_specular,
                    s_r_diffuse=real_diffuse, s_r_specular=real_specular,
                    k_diffuse=None, k_specular=None,
                    k_g_diffuse=None, k_g_specular=None)

        return output


# -------------------------------------------------------------------------------------
#                         END copied and modified by Olivia
# -------------------------------------------------------------------------------------

class KPCN_2ndStrategy(nn.Module):
    """
    use the the features of KPCN and P-buffers to predict a second kernel

    Olivia is working on it. very slowly.

    Args:
        n_in(int): number of input channels in the diffuse/specular streams.
        ksize(int): size of the gather reconstruction kernel.
        depth(int): number of conv layers in each branch for g buffer part.
        width(int): number of feature channels in each branch.
        p_depth(int): number of conv layers in each branch for p buffer part.
    """
    def __init__(self, n_in, ksize=21, depth=9, width=100, p_depth=2, pnet_out=3, use_skip=False):
        super(KPCN_2ndStrategy, self).__init__()
        self.use_skip = use_skip
        self.ksize = ksize
        self.diffuse = ModifiedConvChain(n_in, ksize*ksize, depth=depth, width=width, ksize=5,
            activation="relu", weight_norm=False, pad=False,
            output_type="linear", p_depth=p_depth) 
        self.specular = ModifiedConvChain(n_in, ksize*ksize, depth=depth, width=width, ksize=5,
            activation="relu", weight_norm=False, pad=False,
            output_type="linear", p_depth=p_depth) 
        self.kernel_apply = ops.KernelApply(softmax=True, splat=False)
    def forward(self, data):
        
        # Process the diffuse and specular channels independently
        k_g_diffuse, k_p_diffuse = self.diffuse(data["kpcn_diffuse_in"])
        k_g_specular, k_p_specular = self.specular(data["kpcn_specular_in"])
        
        # G kernel part:

        # Match dimensions
        b_diffuse_g = crop_like(data["kpcn_diffuse_buffer"],
                              k_g_diffuse).contiguous()
        b_specular_g = crop_like(data["kpcn_specular_buffer"],
                               k_g_specular).contiguous()
        # Kernel reconstruction
        r_diffuse_g, _ = self.kernel_apply(b_diffuse_g, k_g_diffuse)
        r_specular_g, _ = self.kernel_apply(b_specular_g, k_g_specular)
        # Combine diffuse/specular/albedo
        albedo_g = crop_like(data["kpcn_albedo"], r_diffuse_g)
        final_specular_g = torch.exp(r_specular_g) - 1
        final_diffuse_g = albedo_g * r_diffuse_g
        final_radiance_g = final_diffuse_g + final_specular_g
        # P kernel part:

        # Match dimensions
        b_diffuse_p = crop_like(data["kpcn_diffuse_buffer"],
                              k_p_diffuse).contiguous()
        b_specular_p = crop_like(data["kpcn_specular_buffer"],
                               k_p_specular).contiguous()
        # Kernel reconstruction
        if self.use_skip:
            k_p_diffuse += k_g_diffuse
            k_p_specular += k_g_specular
        r_diffuse_p, _ = self.kernel_apply(b_diffuse_p, k_p_diffuse)
        r_specular_p, _ = self.kernel_apply(b_specular_p, k_p_specular)
        # Combine diffuse/specular/albedo
        albedo_p = crop_like(data["kpcn_albedo"], r_diffuse_p)
        final_specular_p = torch.exp(r_specular_p) - 1
        final_diffuse_p = albedo_p * r_diffuse_p
        final_radiance_p = final_diffuse_p + final_specular_p

        output = dict(radiance=final_radiance_p, diffuse=r_diffuse_p,
                      specular=r_specular_p, gbuf_radiance=final_radiance_g, gbuf_diffuse=r_diffuse_g,
                      gbuf_specular=r_specular_g, 
                      k_g_specular = k_g_specular, k_g_diffuse = k_g_diffuse, # g kernels
                      k_specular = k_p_specular, k_diffuse = k_p_diffuse) # p kernels
        
        # output = dict(radiance=final_radiance_g, diffuse=r_diffuse_g,
        #               specular=r_specular_g, radiance_p=final_radiance_p, diffuse_p=r_diffuse_p,
        #               specular_p=r_specular_p)

        return output

class SkipKPCN(nn.Module):
    """Re-implementation of [Bako 2017].

    Kernel-Predicting Convolutional Networks for Denoising Monte Carlo
    Renderings: <http://cvc.ucsb.edu/graphics/Papers/SIGGRAPH2017_KPCN/>

    Args:
        n_in(int): number of input channels in the diffuse/specular streams.
        ksize(int): size of the gather reconstruction kernel.
        depth(int): number of conv layers in each branch.
        width(int): number of feature channels in each branch.
    """

    def __init__(self, n_in, ksize=21, depth=9, width=100, pnet_out=3):
        super(SkipKPCN, self).__init__()

        self.ksize = ksize
        self.pnet_size = pnet_out + 2
        gbuf_in = n_in - pnet_out - 2

        self.diffuse = ops.ConvChain(
            n_in, ksize*ksize, depth=depth, width=width, ksize=5,
            activation="relu", weight_norm=False, pad=False,
            output_type="linear")

        self.specular = ops.ConvChain(
            n_in, ksize*ksize, depth=depth, width=width, ksize=5,
            activation="relu", weight_norm=False, pad=False,
            output_type="linear")

        self.gbuf_diffuse = ops.ConvChain(
            gbuf_in, ksize*ksize, depth=depth, width=width, ksize=5,
            activation="relu", weight_norm=False, pad=False,
            output_type="linear")

        self.gbuf_specular = ops.ConvChain(
            gbuf_in, ksize*ksize, depth=depth, width=width, ksize=5,
            activation="relu", weight_norm=False, pad=False,
            output_type="linear")

        self.kernel_apply = ops.KernelApply(softmax=True, splat=False)

    def load_pretrain(self, pth):
        state = torch.load(pth)
        own_state = self.state_dict()
        # load KPCN to gbuf
        for k, v in state.items():
            if k == 'state_dict_dncnn':
                for key, value in v.items():
                    # key, value = v
                    new_key = 'gbuf_'+ key # add 'gbuf' in front of keys
                    if k not in own_state:
                        continue
                    own_state[new_key].copy_(value.data)

        # freeze these parts
        for param in self.gbuf_diffuse.parameters():
            param.requires_grad = False
        for param in self.gbuf_specular.parameters():
            param.requires_grad = False


    def forward(self, data):
        """Forward pass of the model.

        Args:
            data(dict) with keys:
                "kpcn_diffuse_in":
                "kpcn_specular_in":
                "kpcn_diffuse_buffer":
                "kpcn_specular_buffer":
                "kpcn_albedo":

        Returns:
            (dict) with keys:
                "radiance":
                "diffuse":
                "specular":
        """
        # Process the diffuse and specular channels independently
        k_diffuse = self.diffuse(data["kpcn_diffuse_in"])
        k_specular = self.specular(data["kpcn_specular_in"])
            

        # Match dimensions
        b_diffuse = crop_like(data["kpcn_diffuse_buffer"],
                              k_diffuse).contiguous()
        b_specular = crop_like(data["kpcn_specular_buffer"],
                               k_specular).contiguous()
        
        gbuf_final_radiance, gbuf_r_diffuse, gbuf_r_specular = None, None, None
        # Kernel reconstruction
        k_gbuf_diffuse = self.gbuf_diffuse(data["kpcn_diffuse_in"][:,:-self.pnet_size])
        k_gbuf_specular = self.gbuf_specular(data["kpcn_specular_in"][:,:-self.pnet_size])
        # Skip Connection
        # print(k_diffuse.shape)
        k_diffuse += k_gbuf_diffuse
        k_specular += k_gbuf_specular

        gbuf_r_diffuse, _  = self.kernel_apply(b_diffuse, k_gbuf_diffuse)
        gbuf_r_specular, _ = self.kernel_apply(b_specular, k_gbuf_specular)

        albedo = crop_like(data["kpcn_albedo"], gbuf_r_diffuse).contiguous()
        gbuf_final_specular = torch.exp(gbuf_r_specular) - 1
        gbuf_final_diffuse = albedo * gbuf_r_diffuse
        gbuf_final_radiance = gbuf_final_diffuse + gbuf_final_specular


        r_diffuse, _ = self.kernel_apply(b_diffuse, k_diffuse)
        r_specular, _ = self.kernel_apply(b_specular, k_specular)

        # Combine diffuse/specular/albedo
        final_specular = torch.exp(r_specular) - 1
        final_diffuse = albedo * r_diffuse
        final_radiance = final_diffuse + final_specular

        output = dict(radiance=final_radiance, diffuse=r_diffuse,
                    specular=r_specular, gbuf_radiance=gbuf_final_radiance,
                    gbuf_diffuse=gbuf_r_diffuse, gbuf_specular=gbuf_r_specular,
                    k_g_specular = k_gbuf_specular, k_g_diffuse = k_gbuf_diffuse, # g kernels
                    k_specular = k_specular, k_diffuse = k_diffuse) # p kernels (+g)

        return output


class KPCN_Single(nn.Module):
    """Re-implementation of [Bako 2017].

    Kernel-Predicting Convolutional Networks for Denoising Monte Carlo
    Renderings: <http://cvc.ucsb.edu/graphics/Papers/SIGGRAPH2017_KPCN/>

    Args:
        n_in(int): number of input channels in the diffuse/specular streams.
        ksize(int): size of the gather reconstruction kernel.
        depth(int): number of conv layers in each branch.
        width(int): number of feature channels in each branch.
    """

    def __init__(self, n_in, ksize=21, depth=9, width=100):
        super(KPCN_Single, self).__init__()

        self.ksize = ksize

        self.denoiser = ops.ConvChain(
            n_in, ksize*ksize, depth=depth, width=width, ksize=5,
            activation="relu", weight_norm=False, pad=True,
            output_type="linear")

        self.softmax = nn.Softmax(dim=1)

        # self.kernel_apply = ops.KernelApply(softmax=True, splat=False)
        self.kernel_apply = WeightedFilter(channels=3, kernel_size=ksize, bias=False, splat=False)

    def forward(self, data):
        """Forward pass of the model.

        Args:
            data(dict) with keys:
                "kpcn_in":
                "kpcn_albedo":

        Returns:
            (dict) with keys:
                "radiance":
                "diffuse":
                "specular":
        """
        # Process the single channel
        k_denoise = self.denoiser(data["kpcn_in"])

        # Match dimensions
        b_denoise = crop_like(data["kpcn_buffer"],
                              k_denoise).contiguous()

        # Kernel reconstruction
        k_denoise = self.softmax(k_denoise)
        # r_denoise, _ = self.kernel_apply(b_denoise, k_denoise)
        r_denoise = self.kernel_apply(b_denoise, k_denoise)
        
        output = dict(radiance=r_denoise, kernel=k_denoise)

        return output

class PixelDiscriminator(nn.Module):
    def __init__(self, n_in, n_out, use_ch=False):
        super(PixelDiscriminator, self).__init__()
        print('unet', n_in, n_out)
        self.unet = SimpleUNet(n_in, n_out)
        # self.unet = nn.Conv2d(n_in, n_out, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, data, mode='diff'):
        if mode == 'diff': 
            x, _ = self.unet(data['kpcn_diffuse_in'])
            y, _ = self.unet(data['target_diffuse'])
        elif mode =='spec': 
            x, _ = self.unet(data['kpcn_specular_in'])
            y, _ = self.unet(data['target_specular'])
        
        

        output = dict(fake=self.sigmoid(x[:,:1]), real=self.sigmoid(y[:,:1]), feat=x[:,1:])
        return output


###############################################################################
# Create a object with members from a dictionary
###############################################################################

class DictObject:
	def __init__(self, _dict):
		self.__dict__.update(**_dict)

def object_from_dict(_dict):
	return DictObject(_dict)