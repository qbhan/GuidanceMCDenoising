from turtle import forward
from pyparsing import Forward
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
from torch_utils import WeightedFilter

# imports for second stragety
import numpy as np
import torch.nn.functional as F
from support.utils import crop_like
from support.modules import SimpleUNet, ModifiedConvChain, Up, Down, StridedDown, DoubleConv, OutConv
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

        gbuf_r_diffuse, _ = self.kernel_apply(b_diffuse, k_gbuf_diffuse)
        gbuf_r_specular, _ = self.kernel_apply(b_specular, k_gbuf_specular)
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
        
        r_diffuse, _ = self.kernel_apply(b_diffuse, k_diffuse)
        r_specular, _ = self.kernel_apply(b_specular, k_specular)

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

class SingleStreamAdvKPCN(nn.Module):
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
        super(SingleStreamAdvKPCN, self).__init__()

        # TODO check parameters...
        self.ksize = ksize
        self.pnet_size = pnet_out + 2
        self.feat_in = feat_in
        gbuf_in = n_in - pnet_out - 2

        self.p_net = ops.ConvChain(
            self.feat_in, ksize*ksize, depth=depth, width=width, ksize=3,
            activation="relu", weight_norm=False, pad=True,
            output_type="linear")

        self.discriminator = PixelDiscriminator(n_in-7, feat_in+1)

        self.g_net = ops.ConvChain(
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
        for param in self.gbuf.parameters(): ############### TODO was (gbuf_diffuse, gbuf_specular)
            param.requires_grad = False


    def forward(self, data, vis=False):
        """Forward pass of the model.
        Args:
            data(dict) with keys:
                "kpcn_buffer"
                "kpcn_in"
                "paths"
                "target_total"
                "radiance"
                "features"
                #"kpcn_albedo"
                #"target_diffuse"
                #"target_specular"
                
        Returns:
            (dict) with keys:
                "radiance":
                "diffuse":
                "specular":
        """
        # Process the diffuse and specular channels independently
        k_gbuf = self.g_net(data["kpcn_in"][:, :-self.pnet_size])

        buffer = crop_like(data["kpcn_buffer"],
                              k_gbuf).contiguous()

        gbuf_r = self.kernel_apply(buffer, k_gbuf)

        gbuf_final_radiance = torch.exp(gbuf_r) - 1

        # discriminator
        batch = {
            'target': torch.cat([data['target'], data["kpcn_in"][:,10:]], 1), #´TODO?
            'kpcn_in': torch.cat([gbuf_r, crop_like(data["kpcn_in"][:,10:], gbuf_r)], 1), # TODO?
        }
        # dis_diffuse = self.d_diffuse(torch.cat([gbuf_final_diffuse, data["kpcn_diffuse_in"][:,10:]], 1))
        # dis_specular = self.d_specular(torch.cat([gbuf_final_specular, data["kpcn_specular_in"][:,10:]], 1))
        out = self.discriminator(batch, mode='single_stream') # TODO mode???

        fake, real, feat = out['fake'], out['real'], out['feat']

        # Match dimensions
        kernel = self.p_net(feat)
        # k_diffuse = self.p_diffuse(torch.cat((fake_diffuse, feat_diffuse), dim=1))
        # k_specular = self.p_specular(torch.cat((fake_specular, feat_specular), dim=1))
        buffer = crop_like(data["kpcn_buffer"], kernel).contiguous()

        # Skip Connection
        kernel += k_gbuf * fake

        radiance = self.kernel_apply(buffer, self.softmax(kernel))

        final_radiance = torch.exp(radiance) - 1

        if not vis:
            output = dict(radiance=final_radiance, g_radiance=gbuf_final_radiance,
                    s_f=fake, s_r=real,
                    kernel=kernel, g_kernel=k_gbuf)

        else:
            output = dict(radiance=final_radiance, g_radiance=gbuf_final_radiance,
                    s_f=fake, s_r=real,
                    kernel=None, g_kernel=None)

        return output

class NewAdvKPCN_1(nn.Module):
    """Our Implementation of Kernel predicting denoising network using 
    adversarial training.

    each branch uses two independent streams: one for diffuse and one for specular

    Args:
        g_in(int): number of input channels in the gbuf_only streams.
        p_in(int): number of input channels in the pbuf_only streams.
        ksize(int): size of the gather reconstruction kernel.
        depth(int): number of conv layers in each branch.
        width(int): number of feature channels in each branch.
    """

    def __init__(self, g_in, p_in, ksize=21, depth=9, width=100, pnet_out=5, gen_activation="relu", disc_activtion="relu", output_type="linear", strided_down=False, interpolation='kernel'):
        super(NewAdvKPCN_1, self).__init__()

        self.ksize = ksize
        self.pnet_size = pnet_out + 2
        self.interpolation = interpolation

        self.G_diffuse = ops.ConvChain(
            g_in, ksize*ksize, depth=depth, width=width, ksize=5,
            activation="relu", weight_norm=False, pad=False,
            output_type="linear")

        self.G_specular = ops.ConvChain(
            g_in, ksize*ksize, depth=depth, width=width, ksize=5,
            activation="relu", weight_norm=False, pad=False,
            output_type="linear")

        # # should add shallow MLP to substitute simple averaging sample-wise path feature
        # self.p_diff_embed = nn.Sequential(
		# 	nn.Conv2d(p_in * 8, p_in * 8, 1, padding=0),
		# 	nn.ReLU(inplace=True),
		# 	nn.Conv2d(p_in * 8, p_in * 8, 1, padding=0),
		# 	nn.ReLU(inplace=True),
		# 	nn.Conv2d(p_in * 8, p_in, 1, padding=0),
		# 	nn.ReLU(inplace=True),
		# )

        # self.p_spec_embed = nn.Sequential(
		# 	nn.Conv2d(p_in * 8, p_in * 8, 1, padding=0),
		# 	nn.ReLU(inplace=True),
		# 	nn.Conv2d(p_in * 8, p_in * 8, 1, padding=0),
		# 	nn.ReLU(inplace=True),
		# 	nn.Conv2d(p_in * 8, p_in, 1, padding=0),
		# 	nn.ReLU(inplace=True),
		# )

        self.P_diffuse = ops.ConvChain(
            p_in, ksize*ksize, depth=depth, width=width, ksize=5,
            activation="relu", weight_norm=False, pad=False,
            output_type="linear")

        self.P_specular = ops.ConvChain(
            p_in, ksize*ksize, depth=depth, width=width, ksize=5,
            activation="relu", weight_norm=False, pad=False,
            output_type="linear")

        self.dis_in = (g_in - 10) + (p_in - 10) + 3

        self.D_diffuse = PixelDiscriminator(self.dis_in, 1 + 50, strided_down, activation=disc_activtion)
        self.D_specular = PixelDiscriminator(self.dis_in, 1 + 50, strided_down, activation=disc_activtion)

#        self.kernel_apply = ops.KernelApply(softmax=True, splat=False)
        self.kernel_apply = WeightedFilter(channels=3, kernel_size=ksize, bias=False, splat=False)

        if self.interpolation != 'direct':
            self.R_diffuse = ops.ConvChain(
                100, 1, depth=5, width=width, ksize=5,
                activation="relu", weight_norm=False, pad=True,
                output_type="sigmoid")

            self.R_specular = ops.ConvChain(
                100, 1, depth=5, width=width, ksize=5,
                activation="relu", weight_norm=False, pad=True,
                output_type="sigmoid")

        else:
            self.R_diffuse = ops.ConvChain(
                100, ksize*ksize, depth=5, width=width, ksize=5,
                activation="relu", weight_norm=False, pad=True,
                output_type="sigmoid")

            self.R_specular = ops.ConvChain(
                100, ksize*ksize, depth=5, width=width, ksize=5,
                activation="relu", weight_norm=False, pad=True,
                output_type="sigmoid")


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
        # pass path featurs to simple embeddings
        # B, S, C, H, W = data['paths'].shape
        # diff_pbuf = self.p_diff_embed(data['paths'].reshape(B*S, C, H, W))
        # spec_pbuf = self.p_spec_embed(data['paths'].reshape(B*S, C, H, W))

        # new features for path-branch & discriminator
        p_diffuse_in = torch.cat((data["kpcn_diffuse_in"][:,:10], data['paths_diffuse']), dim=1)
        p_specular_in = torch.cat((data["kpcn_specular_in"][:,:10], data['paths_specular']), dim=1)
        dis_feat_diffuse = torch.cat((data["kpcn_diffuse_in"][:,10:], data['paths_diffuse']), dim=1)
        dis_feat_specular = torch.cat((data["kpcn_specular_in"][:,10:], data['paths_specular']), dim=1)

        # p_diffuse_in = torch.cat((data["kpcn_diffuse_in"][:,:10], diff_pbuf), dim=1)
        # p_specular_in = torch.cat((data["kpcn_diffuse_in"][:,:10], spec_pbuf), dim=1)
        # dis_feat = torch.cat((data["kpcn_diffuse_in"][:,:10], data['kpcn_specular_in'], data['paths']), dim=1)

        # Process the diffuse and specular channels independently
        g_k_diffuse = self.G_diffuse(data["kpcn_diffuse_in"]).contiguous()
        g_k_specular = self.G_specular(data["kpcn_specular_in"]).contiguous()
        p_k_diffuse = self.P_diffuse(p_diffuse_in).contiguous()
        p_k_specular = self.P_specular(p_specular_in).contiguous()

        b_diffuse = crop_like(data["kpcn_diffuse_buffer"],
                              g_k_diffuse).contiguous()
        b_specular = crop_like(data["kpcn_specular_buffer"],
                               g_k_specular).contiguous()

        g_r_diffuse = self.kernel_apply(b_diffuse, F.softmax(g_k_diffuse, dim=1))
        g_r_specular = self.kernel_apply(b_specular, F.softmax(g_k_specular, dim=1))
        p_r_diffuse = self.kernel_apply(b_diffuse, F.softmax(p_k_diffuse, dim=1))
        p_r_specular = self.kernel_apply(b_specular, F.softmax(p_k_specular, dim=1))
    

        albedo = crop_like(data["kpcn_albedo"], g_r_diffuse).contiguous()
        g_f_specular = torch.exp(g_r_specular) - 1.0
        g_f_diffuse = albedo * g_r_diffuse
        g_f_radiance = g_f_diffuse + g_f_specular
        p_f_specular = torch.exp(p_r_specular) - 1.0
        p_f_diffuse = albedo * p_r_diffuse
        p_f_radiance = p_f_diffuse + p_f_specular

        # discriminator
        # returns 1 + C channels
        # 1 refers to scoremap
        # C refers to hidden embeddings for kernel interpolation 
        batch_diffuse = {
            'diffuse_target': torch.cat([crop_like(data['target_diffuse'], g_r_diffuse), crop_like(dis_feat_diffuse, g_r_diffuse)], 1),
            'diffuse_G': torch.cat([g_r_diffuse, crop_like(dis_feat_diffuse, g_r_diffuse)], 1),
            'diffuse_P': torch.cat([p_r_diffuse, crop_like(dis_feat_diffuse, p_r_diffuse)], 1),
        }
        batch_specular = {
            'specular_target': torch.cat([crop_like(data['target_specular'], g_r_specular), crop_like(dis_feat_specular, g_r_specular)], 1),
            'specular_G': torch.cat([g_r_specular, crop_like(dis_feat_specular, g_r_specular)], 1),
            'specular_P': torch.cat([p_r_specular, crop_like(dis_feat_specular, p_r_specular)], 1),
        }
        dis_out_diffuse = self.D_diffuse(batch_diffuse, 'newadv1')
        dis_out_specular = self.D_specular(batch_specular, 'newadv1')

        # First attempt
        # use hidden embeddings to find interpolation weight
        # due to sigmoid activation, it will have value between 0 and 1
        int_weight_diffuse = self.R_diffuse(torch.cat([dis_out_diffuse['diffuse_G'][:, 1:], dis_out_diffuse['diffuse_P'][:, 1:]], dim=1))
        int_weight_specular = self.R_specular(torch.cat([dis_out_specular['specular_G'][:, 1:], dis_out_specular['specular_P'][:, 1:]], dim=1))
#        int_weight_diffuse, int_weight_specular = None, None

        if self.interpolation == 'kernel':
            # Interpolate new kernel using the scores
            # This part should be more sophisticated
            # n_k_diffuse = g_k_diffuse * dis_out['diffuse_G'] + p_k_diffuse * dis_out['diffuse_P']
            # n_k_specular = g_k_specular * dis_out['specular_G'] + p_k_specular * dis_out['specular_P']
            n_k_diffuse = g_k_diffuse * int_weight_diffuse + p_k_diffuse * (1.0 - int_weight_diffuse)
            n_k_specular = g_k_specular * int_weight_specular + p_k_specular * (1.0 - int_weight_specular)
            r_diffuse = self.kernel_apply(b_diffuse, F.softmax(n_k_diffuse, dim=1).contiguous())
            r_specular = self.kernel_apply(b_specular, F.softmax(n_k_specular, dim=1).contiguous())

        elif self.interpolation == 'image':
            # Interpolate the images to get the final image
            # in order to reduce redundant computation
            n_k_diffuse, n_k_specular = None, None
            r_diffuse = g_r_diffuse * int_weight_diffuse + p_r_diffuse * (1.0 - int_weight_diffuse)
            r_specular = g_r_specular * int_weight_specular + p_r_specular * (1.0 - int_weight_specular)

        elif self.interpolation == 'direct':
            # use hidden embeddings to predict a new kernel to be applied
            # might need a constraint that can make the new kernel not to diverge
            n_k_diffuse = self.R_diffuse(torch.cat([dis_out_diffuse['diffuse_G'][:, 1:], dis_out_diffuse['diffuse_P'][:, 1:]], dim=1))
            n_k_specular = self.R_specular(torch.cat([dis_out_specular['specular_G'][:, 1:], dis_out_specular['specular_P'][:, 1:]], dim=1))
            r_diffuse = self.kernel_apply(b_diffuse, F.softmax(n_k_diffuse, dim=1).contiguous())
            r_specular = self.kernel_apply(b_specular, F.softmax(n_k_specular, dim=1).contiguous())


        albedo = crop_like(data["kpcn_albedo"], r_diffuse).contiguous()
        final_specular = torch.exp(r_specular) - 1
        final_diffuse = albedo * r_diffuse
        final_radiance = final_diffuse + final_specular

        if not vis:
            output = dict(radiance=final_radiance, diffuse=r_diffuse,
                    specular=r_specular, 
                    g_radiance=g_f_radiance, g_diffuse=g_r_diffuse, 
                    g_specular=g_r_specular,
                    p_radiance=p_f_radiance, p_diffuse=p_r_diffuse, 
                    p_specular=p_r_specular,
                    s_diffuse=dis_out_diffuse['diffuse_target'][:, :1], s_specular=dis_out_specular['specular_target'][:, :1], 
                    s_g_diffuse=dis_out_diffuse['diffuse_G'][:, :1], s_g_specular=dis_out_specular['specular_G'][:, :1],
                    s_p_diffuse=dis_out_diffuse['diffuse_P'][:, :1], s_p_specular=dis_out_specular['specular_P'][:, :1],
                    weight_diffuse=int_weight_diffuse, weight_specular=int_weight_specular,
                    n_k_diffuse=n_k_diffuse, n_k_specular=n_k_specular,
                    g_k_diffuse=g_k_diffuse, g_k_specular=g_k_specular,
                    p_k_diffuse=p_k_diffuse, p_k_specular=p_k_specular)

        else:
            output = dict(radiance=final_radiance, diffuse=r_diffuse,
                    specular=r_specular, 
                    g_radiance=g_f_radiance, g_diffuse=g_r_diffuse, 
                    g_specular=g_r_specular,
                    p_radiance=p_f_radiance, p_diffuse=p_r_diffuse, 
                    p_specular=p_r_specular,
                    s_diffuse=dis_out_diffuse['diffuse_target'][:, :1], s_specular=dis_out_specular['specular_target'][:, :1], 
                    s_g_diffuse=dis_out_diffuse['diffuse_G'][:, :1], s_g_specular=dis_out_specular['specular_G'][:, :1],
                    s_p_diffuse=dis_out_diffuse['diffuse_P'][:, :1], s_p_specular=dis_out_specular['specular_P'][:, :1],
                    weight_diffuse=int_weight_diffuse, weight_specular=int_weight_specular,
                    n_k_diffuse=None, n_k_specular=None,
                    g_k_diffuse=None, g_k_specular=None,
                    p_k_diffuse=None, p_k_specular=None)

        return output


class NewAdvKPCN_2(nn.Module):
    """Our Implementation of Kernel predicting denoising network using 
    adversarial training.

    each branch uses two independent streams: one for diffuse and one for specular

    Args:
        g_in(int): number of input channels in the gbuf_only streams.
        p_in(int): number of input channels in the pbuf_only streams.
        ksize(int): size of the gather reconstruction kernel.
        depth(int): number of conv layers in each branch.
        width(int): number of feature channels in each branch.
    """

    def __init__(self, g_in, p_in, ksize=21, depth=9, width=100, pnet_out=5, disc_activtion="relu", output_type="linear", strided_down=False, interpolation='kernel'):
        super(NewAdvKPCN_2, self).__init__()

        self.ksize = ksize
        self.pnet_size = pnet_out + 2
        self.interpolation = interpolation

        self.G_diffuse = ops.ConvChain(
            g_in, ksize*ksize, depth=depth, width=width, ksize=5,
            activation="relu", weight_norm=False, pad=False,
            output_type="linear")

        self.G_specular = ops.ConvChain(
            g_in, ksize*ksize, depth=depth, width=width, ksize=5,
            activation="relu", weight_norm=False, pad=False,
            output_type="linear")

        # # should add shallow MLP to substitute simple averaging sample-wise path feature
        # self.p_diff_embed = nn.Sequential(
		# 	nn.Conv2d(p_in * 8, p_in * 8, 1, padding=0),
		# 	nn.ReLU(inplace=True),
		# 	nn.Conv2d(p_in * 8, p_in * 8, 1, padding=0),
		# 	nn.ReLU(inplace=True),
		# 	nn.Conv2d(p_in * 8, p_in, 1, padding=0),
		# 	nn.ReLU(inplace=True),
		# )

        # self.p_spec_embed = nn.Sequential(
		# 	nn.Conv2d(p_in * 8, p_in * 8, 1, padding=0),
		# 	nn.ReLU(inplace=True),
		# 	nn.Conv2d(p_in * 8, p_in * 8, 1, padding=0),
		# 	nn.ReLU(inplace=True),
		# 	nn.Conv2d(p_in * 8, p_in, 1, padding=0),
		# 	nn.ReLU(inplace=True),
		# )

        self.P_diffuse = ops.ConvChain(
            p_in, ksize*ksize, depth=depth, width=width, ksize=5,
            activation="relu", weight_norm=False, pad=False,
            output_type="linear")

        self.P_specular = ops.ConvChain(
            p_in, ksize*ksize, depth=depth, width=width, ksize=5,
            activation="relu", weight_norm=False, pad=False,
            output_type="linear")

        self.dis_in = (g_in - 10) + (p_in - 10) + 3

        if self.interpolation == 'direct': outc = ksize*ksize
        else: outc = 1
        self.D_diffuse = PixelClassifier(self.dis_in, outc=outc, strided_down=strided_down, activation=disc_activtion)
        self.D_specular = PixelClassifier(self.dis_in, outc=outc, strided_down=strided_down, activation=disc_activtion)

#        self.kernel_apply = ops.KernelApply(softmax=True, splat=False)
        self.kernel_apply = WeightedFilter(channels=3, kernel_size=ksize, bias=False, splat=False)

    def forward(self, data, mode='train'):
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
        # pass path featurs to simple embeddings
        # B, S, C, H, W = data['paths'].shape
        # diff_pbuf = self.p_diff_embed(data['paths'].reshape(B*S, C, H, W))
        # spec_pbuf = self.p_spec_embed(data['paths'].reshape(B*S, C, H, W))

        # new features for path-branch & discriminator
        p_diffuse_in = torch.cat((data["kpcn_diffuse_in"][:,:10], data['paths_diffuse']), dim=1)
        p_specular_in = torch.cat((data["kpcn_specular_in"][:,:10], data['paths_specular']), dim=1)
        dis_feat_diffuse = torch.cat((data["kpcn_diffuse_in"][:,10:], data['paths_diffuse']), dim=1)
        dis_feat_specular = torch.cat((data["kpcn_specular_in"][:,10:], data['paths_specular']), dim=1)

        # p_diffuse_in = torch.cat((data["kpcn_diffuse_in"][:,:10], diff_pbuf), dim=1)
        # p_specular_in = torch.cat((data["kpcn_diffuse_in"][:,:10], spec_pbuf), dim=1)
        # dis_feat = torch.cat((data["kpcn_diffuse_in"][:,:10], data['kpcn_specular_in'], data['paths']), dim=1)

        # Process the diffuse and specular channels independently
        g_k_diffuse = self.G_diffuse(data["kpcn_diffuse_in"]).contiguous()
        g_k_specular = self.G_specular(data["kpcn_specular_in"]).contiguous()
        p_k_diffuse = self.P_diffuse(p_diffuse_in).contiguous()
        p_k_specular = self.P_specular(p_specular_in).contiguous()

        b_diffuse = crop_like(data["kpcn_diffuse_buffer"],
                              g_k_diffuse).contiguous()
        b_specular = crop_like(data["kpcn_specular_buffer"],
                               g_k_specular).contiguous()

        g_r_diffuse = self.kernel_apply(b_diffuse, F.softmax(g_k_diffuse, dim=1))
        g_r_specular = self.kernel_apply(b_specular, F.softmax(g_k_specular, dim=1))
        p_r_diffuse = self.kernel_apply(b_diffuse, F.softmax(p_k_diffuse, dim=1))
        p_r_specular = self.kernel_apply(b_specular, F.softmax(p_k_specular, dim=1))
    

        albedo = crop_like(data["kpcn_albedo"], g_r_diffuse).contiguous()
        g_f_specular = torch.exp(g_r_specular) - 1.0
        g_f_diffuse = albedo * g_r_diffuse
        g_f_radiance = g_f_diffuse + g_f_specular
        p_f_specular = torch.exp(p_r_specular) - 1.0
        p_f_diffuse = albedo * p_r_diffuse
        p_f_radiance = p_f_diffuse + p_f_specular

        # discriminator
        # returns 1 + C channels
        # 1 refers to scoremap
        # C refers to hidden embeddings for kernel interpolation 
        batch_diffuse = {
            'clean': torch.cat([crop_like(data['target_diffuse'], g_r_diffuse), crop_like(dis_feat_diffuse, g_r_diffuse)], 1),
            'noisy_G': torch.cat([g_r_diffuse, crop_like(dis_feat_diffuse, g_r_diffuse)], 1),
            'noisy_P': torch.cat([p_r_diffuse, crop_like(dis_feat_diffuse, p_r_diffuse)], 1),
        }
        batch_specular = {
            'clean': torch.cat([crop_like(data['target_specular'], g_r_specular), crop_like(dis_feat_specular, g_r_specular)], 1),
            'noisy_G': torch.cat([g_r_specular, crop_like(dis_feat_specular, g_r_specular)], 1),
            'noisy_P': torch.cat([p_r_specular, crop_like(dis_feat_specular, p_r_specular)], 1),
        }
        dis_out_diffuse = self.D_diffuse(batch_diffuse, mode)
        dis_out_specular = self.D_specular(batch_specular, mode)

        
        if self.interpolation == 'kernel':
            int_weight_diffuse, int_weight_specular = dis_out_diffuse['weight'], dis_out_specular['weight']
            n_k_diffuse = g_k_diffuse * int_weight_diffuse + p_k_diffuse * (1.0 - int_weight_diffuse)
            n_k_specular = g_k_specular * int_weight_specular + p_k_specular * (1.0 - int_weight_specular)
            r_diffuse = self.kernel_apply(b_diffuse, F.softmax(n_k_diffuse, dim=1).contiguous())
            r_specular = self.kernel_apply(b_specular, F.softmax(n_k_specular, dim=1).contiguous())
        elif self.interpolation == 'direct':
            # use hidden embeddings to predict a new kernel to be applied
            # might need a constraint that can make the new kernel not to diverge
            int_weight_diffuse, int_weight_specular = None, None
            n_k_diffuse, n_k_specular = dis_out_diffuse['weight'], dis_out_specular['weight']
            r_diffuse = self.kernel_apply(b_diffuse, F.softmax(n_k_diffuse, dim=1).contiguous())
            r_specular = self.kernel_apply(b_specular, F.softmax(n_k_specular, dim=1).contiguous())
        elif self.interpolation == 'image':
            int_weight_diffuse, int_weight_specular = dis_out_diffuse['weight'], dis_out_specular['weight']
            n_k_diffuse, n_k_specular = None, None
            r_diffuse = g_r_diffuse * int_weight_diffuse + p_r_diffuse * (1.0 - int_weight_diffuse)
            r_specular = g_r_specular * int_weight_specular + p_r_specular * (1.0 - int_weight_specular)
            
        albedo = crop_like(data["kpcn_albedo"], r_diffuse).contiguous()
        final_specular = torch.exp(r_specular) - 1.0
        final_diffuse = albedo * r_diffuse
        final_radiance = final_diffuse + final_specular
        if mode == 'vis':
            output = dict(radiance=final_radiance, diffuse=r_diffuse,
                    specular=r_specular, 
                    g_radiance=g_f_radiance, g_diffuse=g_r_diffuse, 
                    g_specular=g_r_specular,
                    p_radiance=p_f_radiance, p_diffuse=p_r_diffuse, 
                    p_specular=p_r_specular,
                    s_diffuse=dis_out_diffuse['clean'], s_specular=dis_out_specular['clean'], 
                    s_g_diffuse=dis_out_diffuse['noisy_G'], s_g_specular=dis_out_specular['noisy_G'],
                    s_p_diffuse=dis_out_diffuse['noisy_P'], s_p_specular=dis_out_specular['noisy_P'],
                    weight_diffuse=int_weight_diffuse, weight_specular=int_weight_specular,
                    n_k_diffuse=n_k_diffuse, n_k_specular=n_k_specular,
                    g_k_diffuse=g_k_diffuse, g_k_specular=g_k_specular,
                    p_k_diffuse=p_k_diffuse, p_k_specular=p_k_specular)

        elif mode == 'test':
            output = dict(radiance=final_radiance, diffuse=r_diffuse,
                    specular=r_specular, 
                    g_radiance=g_f_radiance, g_diffuse=g_r_diffuse, 
                    g_specular=g_r_specular,
                    p_radiance=p_f_radiance, p_diffuse=p_r_diffuse, 
                    p_specular=p_r_specular,
                    s_diffuse=dis_out_diffuse['clean'], s_specular=dis_out_specular['clean'], 
                    s_g_diffuse=dis_out_diffuse['noisy_G'], s_g_specular=dis_out_specular['noisy_G'],
                    s_p_diffuse=dis_out_diffuse['noisy_P'], s_p_specular=dis_out_specular['noisy_P'],
                    weight_diffuse=int_weight_diffuse, weight_specular=int_weight_specular,
                    n_k_diffuse=None, n_k_specular=None,
                    g_k_diffuse=None, g_k_specular=None,
                    p_k_diffuse=None, p_k_specular=None)
        elif mode == 'train':
            output = dict(radiance=final_radiance, diffuse=r_diffuse,
                    specular=r_specular, 
                    g_radiance=g_f_radiance, g_diffuse=g_r_diffuse, 
                    g_specular=g_r_specular,
                    p_radiance=p_f_radiance, p_diffuse=p_r_diffuse, 
                    p_specular=p_r_specular,
                    s_diffuse=dis_out_diffuse['clean'], s_specular=dis_out_specular['clean'], 
                    s_g_diffuse=dis_out_diffuse['noisy_G'], s_g_specular=dis_out_specular['noisy_G'],
                    s_p_diffuse=dis_out_diffuse['noisy_P'], s_p_specular=dis_out_specular['noisy_P'],
                    )

        return output

class ModKPCN(nn.Module):
    """Re-implementation of [Bako 2017].

    Kernel-Predicting Convolutional Networks for Denoising Monte Carlo
    Renderings: <http://cvc.ucsb.edu/graphics/Papers/SIGGRAPH2017_KPCN/>

    Args:
        n_in(int): number of input channels in the diffuse/specular streams.
        ksize(int): size of the gather reconstruction kernel.
        depth(int): number of conv layers in each branch.
        width(int): number of feature channels in each branch.
    """

    def __init__(self, g_in, p_in, ksize=21, depth=18, width=100, activation="relu", output_type="linear"):
        super(ModKPCN, self).__init__()

        self.ksize = ksize

        self.G = ops.ConvChain(
            g_in, ksize*ksize, depth=depth, width=width, ksize=3,
            activation=activation, weight_norm=False, pad=True,
            output_type=output_type)

        self.P = ops.ConvChain(
            p_in, ksize*ksize, depth=depth, width=width, ksize=3,
            activation=activation, weight_norm=False, pad=True,
            output_type=output_type)

        self.kernel_apply = ops.KernelApply(softmax=True, splat=True)

    def forward(self, data, vis=False):
        # new features for path-branch & discriminator
        p_in = torch.cat([data['kpcn_in'][:, :20], data['paths'].mean(1)], dim=1) # currently using mean of path features over samples. Might further use path model for better compression

        # Denoise both gbuf_only & pbuf_only branch
        g_kernel = self.G(data["kpcn_in"])
        p_kernel = self.P(p_in)

        assert g_kernel.shape[1] == p_kernel.shape[1]

        buffer = crop_like(data["kpcn_buffer"],
                              g_kernel).contiguous()

        g_rad, _ = self.kernel_apply(buffer, g_kernel)
        p_rad, _ = self.kernel_apply(buffer, p_kernel)

        if vis:
            return dict(g_radiance=g_rad, p_radiance=p_rad,
                        g_kernel=g_kernel, p_kernel=p_kernel)
        else:
            return dict(g_radiance=g_rad, p_radiance=p_rad,
                        g_kernel=None, p_kernel=None)

class SingleKPCN(nn.Module):
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
        super(SingleKPCN, self).__init__()

        self.ksize = ksize

        self.denoise = ops.ConvChain(
            n_in, ksize*ksize, depth=depth, width=width, ksize=5,
            activation="relu", weight_norm=False, pad=False,
            output_type="linear")

        self.kernel_apply = ops.KernelApply(softmax=True, splat=True)

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
        k_denoise = self.diffuse(data["kpcn_in"])

        # Match dimensions
        b_denoise = crop_like(data["kpcn_buffer"],
                              k_denoise).contiguous()
        # Kernel reconstruction
        r_denoise, _ = self.kernel_apply(b_denoise, k_denoise)

        # Combine diffuse/specular/albedo
        # albedo = crop_like(data["kpcn_albedo"], r_diffuse)
        # final_specular = th.exp(r_specular) - 1
        # final_diffuse = albedo * r_diffuse
        # final_radiance = final_diffuse + final_specular
        if vis: output = dict(radiance=r_denoise, k_denoise=k_denoise)
        else: output = dict(radiance=r_denoise, k_denoise=None)

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

        self.kernel_apply = ops.KernelApply(softmax=True, splat=False)

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
    def __init__(self, n_in, n_out, use_ch=False, strided_down=False, activation="relu"):
        super(PixelDiscriminator, self).__init__()
        # print('unet', n_in, n_out)
        # print("PixelDiscriminator", activation)
        self.unet = SimpleUNet(n_in, n_out, strided_down=strided_down, activation=activation)
        # self.unet = nn.Conv2d(n_in, n_out, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, data, mode='diff'):
        if mode == 'diff': 
            x, _ = self.unet(data['kpcn_diffuse_in'])
            y, _ = self.unet(data['target_diffuse'])
            output = dict(fake=self.sigmoid(x[:,:1]), real=self.sigmoid(y[:,:1]), feat=x[:,1:], fake_2=None)
        elif mode =='spec': 
            x, _ = self.unet(data['kpcn_specular_in'])
            y, _ = self.unet(data['target_specular'])
            output = dict(fake=self.sigmoid(x[:,:1]), real=self.sigmoid(y[:,:1]), feat=x[:,1:], fake_2=None)
        elif mode == 'adv_2':
            x, _ = self.unet(data['g_rad_in'])
            y, _ = self.unet(data['target_in'])
            z, _ = self.unet(data['p_rad_in'])
            # output = dict(fake=self.sigmoid(x[:,:1]), real=self.sigmoid(y[:,:1]), feat=x[:,1:], fake_2=self.sigmoid(z[:,:1]))
            output = dict(fake=self.sigmoid(x), real=self.sigmoid(y), feat=x[:,1:], fake_2=self.sigmoid(z))
        elif mode == 'single_stream':
            x, _ = self.unet(data['kpcn_in'])
            y, _ = self.unet(data['target'])
            z, _ = torch.zeros_like(y) # TODO add this line in git
            output = dict(fake=self.sigmoid(x[:,:1]), real=self.sigmoid(y[:,:1]), feat=x[:,1:], fake_2=None)
        elif mode == 'final':
            x, _ = self.unet(data['final_in'])
            output = dict(fake=self.sigmoid(x[:,:1]))
        else:
            output = {}
            for k in data:
                x, _ = self.unet(data[k])
                x = self.sigmoid(x)
                output[k] = x

        return output


class PixelClassifier(nn.Module):
    '''
    A pixel-wise classifier with a U-Net like architecture.
    Consisted of two decoders sharing the same decoder.
    Each decoder outputs 1. Score Map and 2. Interpolation Weight.
    The second decoder will have concatenated embeddings of each pass of encoder.
    '''
    def __init__(self, ic, outc=1, intermc=64, bilinear=True, strided_down=False, activation='leaky_relu'):
        super(PixelClassifier, self).__init__()
        self.ic = ic
        self.intermc = intermc
        self.outc = outc
        self.factor = 2 if bilinear else 1

        if strided_down:
            down = StridedDown
        else:
            down = Down

        # encoder
        self.enc_1 = DoubleConv(ic, intermc, activation=activation)
        self.enc_2 = down(intermc, intermc, activation=activation)
        self.enc_3 = down(intermc, intermc*2, activation=activation)
        self.enc_4 = down(intermc*2, intermc*4 // self.factor, activation=activation)

        # decoder 1
        # decoder for returning scoremap
        self.dec1_1 = Up(intermc*4, intermc*2 // self.factor, bilinear, activation=activation)
        self.dec1_2 = Up(intermc*2, intermc, bilinear, activation=activation)
        self.dec1_3 = Up(intermc*2, intermc, bilinear, activation=activation)
        self.dec1_4 = OutConv(intermc, 1, activation='sigmoid')

        # decoder 2
        # decoder for returning interpolation weight
        self.dec2_1 = Up(intermc*8, intermc*2, bilinear, activation=activation)
        self.dec2_2 = Up(intermc*4, intermc, bilinear, activation=activation)
        self.dec2_3 = Up(intermc*3, intermc, bilinear, activation=activation)
        self.dec2_4 = OutConv(intermc, outc, activation='sigmoid')

    def __str__(self):
        return "Pixel-wise Classifier i{}in{}o{}".format(self.ic, self.intermc, self.outc)
    
    def forward(self, data, mode='train'):
        # need to put assertion for keys in data input
        assert 'noisy_G' in data
        assert 'noisy_P' in data
        assert 'clean' in data
        output = {}
        encode = {}
        for k in data:
            enc1 = self.enc_1(data[k])  # c=64
            enc2 = self.enc_2(enc1)     # c=64
            enc3 = self.enc_3(enc2)     # c=128
            enc4 = self.enc_4(enc3)     # c=128
            if mode == 'train':
                out = self.dec1_1(enc4, enc3)   # c=64
                out = self.dec1_2(out, enc2)    # c=64
                out = self.dec1_3(out, enc1)    # c=64
                out = self.dec1_4(out)          # c=64
                output[k] = out
            else:
                output[k] = None
            encode[k] = [enc4, enc3, enc2, enc1]

        out = self.dec2_1(torch.cat([encode['noisy_G'][0], encode['noisy_G'][0]], dim=1), torch.cat([encode['noisy_G'][1], encode['noisy_G'][1]], dim=1))
        out = self.dec2_2(out, torch.cat([encode['noisy_G'][2], encode['noisy_P'][2]], dim=1))
        out = self.dec2_3(out, torch.cat([encode['noisy_G'][3], encode['noisy_P'][3]], dim=1))
        out = self.dec2_4(out)
        output['weight'] = out
        return output


###############################################################################
# Create a object with members from a dictionary
###############################################################################

class DictObject:
	def __init__(self, _dict):
		self.__dict__.update(**_dict)

def object_from_dict(_dict):
	return DictObject(_dict)
