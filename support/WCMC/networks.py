import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import ConvChain, Autoencoder
from torch_utils import WeightedFilter
from .utils import crop_like

class PathNet(nn.Module):
    """Path embedding network (Cho et al. 2021)
    """

    def __init__(self, ic, intermc=64, outc=3):
        super(PathNet, self).__init__()
        self.ic = ic
        self.intermc = intermc
        self.outc = outc
        self.final_ic = intermc + intermc

        self.embedding = ConvChain(ic, intermc, width=intermc, depth=3,
                ksize=1, pad=False)
        self.propagation = Autoencoder(intermc, intermc, num_levels=3, 
                increase_factor=2.0, num_convs=3, width=intermc, ksize=3,
                output_type="leaky_relu", pooling="max")
        self.final = ConvChain(self.final_ic, outc, width=self.final_ic, 
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


class KPCN(nn.Module):
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
        super(KPCN, self).__init__()

        self.ksize = ksize

        self.diffuse = ConvChain(
            n_in, ksize*ksize, depth=depth, width=width, ksize=5,
            activation="relu", weight_norm=False, pad=False,
            output_type="linear")

        self.specular = ConvChain(
            n_in, ksize*ksize, depth=depth, width=width, ksize=5,
            activation="relu", weight_norm=False, pad=False,
            output_type="linear")

        self.kernel_apply = WeightedFilter(channels=3, kernel_size=ksize, bias=False, splat=False)

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
        k_diffuse = self.diffuse(data["kpcn_diffuse_in"]).contiguous()
        k_specular = self.specular(data["kpcn_specular_in"]).contiguous()

        # Match dimensions
        b_diffuse = crop_like(data["kpcn_diffuse_buffer"],
                              k_diffuse).contiguous()
        b_specular = crop_like(data["kpcn_specular_buffer"],
                               k_specular).contiguous()

        # Kernel reconstruction
        r_diffuse = self.kernel_apply(b_diffuse, F.softmax(k_diffuse, dim=1))
        r_specular = self.kernel_apply(b_specular, F.softmax(k_specular, dim=1))

        # Combine diffuse/specular/albedo
        albedo = crop_like(data["kpcn_albedo"], r_diffuse)
        final_specular = torch.exp(r_specular) - 1
        final_diffuse = albedo * r_diffuse
        final_radiance = final_diffuse + final_specular

        output = dict(radiance=final_radiance, diffuse=r_diffuse,
                      specular=r_specular)

        return output