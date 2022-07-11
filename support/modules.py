import torch
import torch.nn as nn
import torch.nn.functional as F
from support.utils import crop_like

class SingleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, activation="relu"):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        
        if activation == "linear":
            act_fn = nn.Identity()
        elif activation == "relu":
            act_fn =  nn.ReLU(inplace=True)
        elif activation == "leaky_relu":
            act_fn = nn.LeakyReLU(inplace=True)
        else:
            raise ValueError("Unknon output type '{}'".format(activation))

        
        # print(activation)
        # print(act_fn)
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            act_fn, #nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            act_fn # nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, activation="relu"):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            # nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels, activation=activation)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class StridedDown(nn.Module):
    """Downscaling with strided conv"""
    def __init__(self, in_channels, out_channels, activation="relu"):
        super().__init__()
        self.strided_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2),         # TODO fix!!!
            DoubleConv(in_channels, out_channels, activation=activation)
        )

    def forward(self, x):
        return self.strided_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, activation="relu"):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, activation=activation)
        else:
            self.up = nn.Upsample(scale_factor=2, mode='nearest')
            self.conv = DoubleConv(in_channels, out_channels, activation=activation)

    def forward(self, x1, x2):
        # print('up', x1.shape)
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class SimpleUNet(nn.Module):
    def __init__(self, n_channels, n_classes, hidden=64, bilinear=True, strided_down=False, activation="relu"):
        super(SimpleUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        factor = 2 if bilinear else 1

        self.inc = DoubleConv(n_channels, hidden, activation=activation)
        
        if strided_down:
            down = StridedDown
        else:
            down = Down
        
        self.down1 = down(hidden, hidden, activation=activation)
        self.down2 = down(hidden, hidden*2, activation=activation)
        self.down3 = down(hidden*2, hidden*4, activation=activation)
        self.down4 = down(hidden*4, hidden*8 // factor, activation=activation)

        self.up1 = Up(hidden*8, hidden*4 // factor, bilinear, activation=activation)
        self.up2 = Up(hidden*4, hidden*2 // factor, bilinear, activation=activation)
        # self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(hidden*2, hidden, bilinear, activation=activation)
        self.up4 = Up(hidden*2, hidden, bilinear, activation=activation)
        self.outc = OutConv(hidden, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        # print('inc x1', x1.shape)
        x2 = self.down1(x1)
        # print('down x2', x2.shape)
        x3 = self.down2(x2)
        # print('down x3', x3.shape)
        x4 = self.down3(x3)
        # print('down x4', x4.shape)
        x5 = self.down4(x4)
        # print('down x5', x5.shape)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        # print('up x3', x.shape)
        x = self.up3(x, x2)
        # print('up x2', x.shape)
        x = self.up4(x, x1)
        # print('up x1', x.shape)
        x = self.outc(x)
        return x, x5
    
    
class RecurrentBlock(nn.Module):
    
    def __init__(self, input_nc, output_nc, device, downsampling=False, bottleneck=False, upsampling=False):
        super(RecurrentBlock, self).__init__()

        self.input_nc = input_nc
        self.output_nc = output_nc

        self.downsampling = downsampling
        self.upsampling = upsampling
        self.bottleneck = bottleneck

        self.hidden = None
        self.device = device

        if self.downsampling:
            self.l1 = nn.Sequential(
                    nn.Conv2d(input_nc, output_nc, 3, padding=1),
                    nn.LeakyReLU(negative_slope=0.1)
                )
            self.l2 = nn.Sequential(
                    nn.Conv2d(2 * output_nc, output_nc, 3, padding=1),
                    nn.LeakyReLU(negative_slope=0.1),
                    nn.Conv2d(output_nc, output_nc, 3, padding=1),
                    nn.LeakyReLU(negative_slope=0.1),
                )
        elif self.upsampling:
            self.l1 = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.Conv2d(2 * input_nc, output_nc, 3, padding=1),
                    nn.LeakyReLU(negative_slope=0.1),
                    nn.Conv2d(output_nc, output_nc, 3, padding=1),
                    nn.LeakyReLU(negative_slope=0.1),
                )
        elif self.bottleneck:
            self.l1 = nn.Sequential(
                    nn.Conv2d(input_nc, output_nc, 3, padding=1),
                    nn.LeakyReLU(negative_slope=0.1)
                )
            self.l2 = nn.Sequential(
                    nn.Conv2d(2 * output_nc, output_nc, 3, padding=1),
                    nn.LeakyReLU(negative_slope=0.1),
                    nn.Conv2d(output_nc, output_nc, 3, padding=1),
                    nn.LeakyReLU(negative_slope=0.1),
                )

    def forward(self, inp):

        if self.downsampling:
            op1 = self.l1(inp)
            op2 = self.l2(torch.cat((op1, self.hidden), dim=1))

            self.hidden = op2

            return op2
        elif self.upsampling:
            op1 = self.l1(inp)

            return op1
        elif self.bottleneck:
            op1 = self.l1(inp)
            op2 = self.l2(torch.cat((op1, self.hidden), dim=1))

            self.hidden = op2

            return op2

    def reset_hidden(self, inp, dfac):
        size = list(inp.size())
        size[1] = self.output_nc
        size[2] /= dfac
        size[3] /= dfac
        size = [ int(x) for x in size]
        self.hidden_size = size
        # print(size)
        self.hidden = torch.zeros(*(size)).to(self.device)
  
  
class RecurrentAE(nn.Module):

    def __init__(self, input_nc, device, output_nc = 5):
        super(RecurrentAE, self).__init__()

        self.d1 = RecurrentBlock(input_nc=input_nc, output_nc=32, device=device, downsampling=True)
        self.d2 = RecurrentBlock(input_nc=32, output_nc=43, device=device, downsampling=True)
        self.d3 = RecurrentBlock(input_nc=43, output_nc=57, device=device, downsampling=True)
        self.d4 = RecurrentBlock(input_nc=57, output_nc=76, device=device, downsampling=True)
        self.d5 = RecurrentBlock(input_nc=76, output_nc=101, device=device, downsampling=True)

        self.bottleneck = RecurrentBlock(input_nc=101, output_nc=101, device=device, bottleneck=True)
        # self.bottleneck = RecurrentBlock(input_nc=57, output_nc=57, device=device, bottleneck=True)

        self.u5 = RecurrentBlock(input_nc=101, output_nc=76, device=device, upsampling=True)
        self.u4 = RecurrentBlock(input_nc=76, output_nc=57, device=device, upsampling=True)
        self.u3 = RecurrentBlock(input_nc=57, output_nc=43, device=device, upsampling=True)
        self.u2 = RecurrentBlock(input_nc=43, output_nc=32, device=device, upsampling=True)
        self.u1 = RecurrentBlock(input_nc=32, output_nc=output_nc, device=device, upsampling=True)
        self.output_nc = output_nc

    def set_input(self, inp):
        self.inp = inp

    def forward(self):
        d1 = F.max_pool2d(input=self.d1(self.inp), kernel_size=2)
        d2 = F.max_pool2d(input=self.d2(d1), kernel_size=2)
        d3 = F.max_pool2d(input=self.d3(d2), kernel_size=2)
        d4 = F.max_pool2d(input=self.d4(d3), kernel_size=2)
        d5 = F.max_pool2d(input=self.d5(d4), kernel_size=2)

        b = self.bottleneck(d5)
        # b = self.bottleneck(d3)

        u5 = self.u5(torch.cat((b, d5), dim=1))
        u4 = self.u4(torch.cat((u5, d4), dim=1))
        u3 = self.u3(torch.cat((u4, d3), dim=1))
        # u3 = self.u3(torch.cat((b, d3), dim=1))
        u2 = self.u2(torch.cat((u3, d2), dim=1))
        u1 = self.u1(torch.cat((u2, d1), dim=1))

        return u1

    def reset_hidden(self):
        self.d1.reset_hidden(self.inp, dfac=1)
        self.d2.reset_hidden(self.inp, dfac=2)
        self.d3.reset_hidden(self.inp, dfac=4)
        self.d4.reset_hidden(self.inp, dfac=8)
        self.d5.reset_hidden(self.inp, dfac=16)

        self.bottleneck.reset_hidden(self.inp, dfac=32)
        # self.bottleneck.reset_hidden(self.inp, dfac=8)

        self.u5.reset_hidden(self.inp, dfac=16)
        self.u4.reset_hidden(self.inp, dfac=8)
        self.u3.reset_hidden(self.inp, dfac=4)
        self.u2.reset_hidden(self.inp, dfac=2)
        self.u1.reset_hidden(self.inp, dfac=1)

# -------------------------------------------------------------------------------------
#                         START copied and modified by Olivia
# -------------------------------------------------------------------------------------
class ModifiedConvChain(nn.Module):
    """A simple stack of convolution layers.

    Args:
        ninputs(int): number of input channels.
        noutputs(int): number of output channels.
        ksize(int): size of all the convolution kernels.
        width(int): number of channels per intermediate layer.
        depth(int): number of intermadiate layers.
        stride(int): stride of the convolution.
        pad(bool): if True, maintains spatial resolution by 0-padding,
            otherwise keep only the valid part.
        normalize(bool): applies normalization if True.
        normalization_type(str): either batch or instance.
        output_type(str): one of linear, relu, leaky_relu, tanh, elu.
        activation(str): one of relu, leaky_relu, tanh, elu.
        weight_norm(bool): applies weight normalization if True.
    """
    def __init__(self, ninputs, noutputs, ksize=3, width=64, depth=3, stride=1,
                 pad=True, normalize=False, normalization_type="batch",
                 output_type="linear", activation="relu", weight_norm=True, p_depth=3, pnet_out=5,
                 use_skip=False):
        super(ModifiedConvChain, self).__init__()

        self.use_skip = use_skip
        if depth <= 0:
            raise ValueError("negative network depth.")

        if pad:
            padding = ksize//2
        else:
            padding = 0

        self.layers = []
        for d in range(depth-1):
            if d == 0:
                _in = ninputs
            else:
                _in = width
            self.layers.append(
                ModifiedConvChain._ConvBNRelu(_in, ksize, width, normalize=normalize,
                                      normalization_type=normalization_type,
                                      padding=padding, stride=stride,
                                      activation=activation,
                                      weight_norm=weight_norm))

        # Rename layers
        for im, m in enumerate(self.layers):
            if im == len(self.layers)-1:
                name = "prediction"
            else:
                name = "layer_{}".format(im)
            self.add_module(name, m)
        
        # Last layer
        if depth > 1:
            _in = width
        else:
            _in = ninputs

        conv = nn.Conv2d(_in, noutputs, ksize, bias=True, padding=padding)
        if weight_norm:
            conv = nn.utils.weight_norm(conv)
        conv.bias.data.zero_()
        if output_type == "elu" or output_type == "softplus":
            nn.init.xavier_uniform_(
                conv.weight.data, nn.init.calculate_gain("relu"))
        else:
            nn.init.xavier_uniform_(
                conv.weight.data, nn.init.calculate_gain(output_type))
        self.last_layer_gbuf = conv

        conv = nn.Conv2d(_in, noutputs, ksize, bias=True, padding=padding)
        if weight_norm:
            conv = nn.utils.weight_norm(conv)
        conv.bias.data.zero_()
        if output_type == "elu" or output_type == "softplus":
            nn.init.xavier_uniform_(
                conv.weight.data, nn.init.calculate_gain("relu"))
        else:
            nn.init.xavier_uniform_(
                conv.weight.data, nn.init.calculate_gain(output_type))
        self.last_layer_pbuf = conv


        # last activation
        if output_type == "linear":
            self.last_activation = nn.Identity()
        elif output_type == "relu":
            self.last_activation =  nn.ReLU(inplace=True)
        elif output_type == "leaky_relu":
            self.last_activation =  nn.LeakyReLU(inplace=True)
        elif output_type == "sigmoid":
            self.last_activation =  nn.Sigmoid()
        elif output_type == "tanh":
            self.last_activation =  nn.Tanh()
        elif output_type == "elu":
            self.last_activation =  nn.ELU()
        elif output_type == "softplus":
            self.last_activation =  nn.Softplus()
        else:
            raise ValueError("Unknon output type '{}'".format(output_type))

        # P-buffer network part
        self.p_layers = nn.ModuleList()

        for i in range(p_depth):
            if i == 0:
                n_in = width+pnet_out
            else:
                n_in = width
            p_conv =  ModifiedConvChain._ConvBNRelu(n_in, 3, width, normalize=normalize,
                                      normalization_type=normalization_type,
                                      padding=1, stride=1,
                                      activation=activation,
                                      weight_norm=weight_norm)
            self.p_layers.append(p_conv)

    def forward(self, x, gbuf_size=34):
        # G-buffer part
        g_features = x[:,:gbuf_size]
        # for m in self.children():
        for m in self.layers:
            # print(g_features.shape)
            g_features = m(g_features)

        # last layer
        # print('kernel prediction')
        g_kernel = self.last_layer_gbuf(g_features)
        g_kernel = self.last_activation(g_kernel)
    	
        # P-buffer part
        # TODO
        # iterate over a few conv layers with (p_buffer input and g_features)
        # return the result together with g_kernel
        #p_features = g_features + p_buffer #### ? + p buffers??
        p_buffer = crop_like(x[:,gbuf_size:], g_features)
        p_features = torch.cat((g_features, p_buffer), dim=1) #### TODO check dimension
        for layer in self.p_layers:
            p_features = layer(p_features)
        
        # last layer
        p_kernel = self.last_layer_pbuf(p_features)
        p_kernel = self.last_activation(p_kernel)

        if self.use_skip:
            p_kernel += g_kernel

        return g_kernel, p_kernel

    class _ConvBNRelu(nn.Module):
        """Helper class that implements a simple Conv-(Norm)-Activation group.

        Args:
            ninputs(int): number of input channels.
            ksize(int): size of all the convolution kernels.
            noutputs(int): number of output channels.
            stride(int): stride of the convolution.
            pading(int): amount of 0-padding.
            normalize(bool): applies normalization if True.
            normalization_type(str): either batch or instance.
            activation(str): one of relu, leaky_relu, tanh, elu.
            weight_norm(bool): if True applies weight normalization.
        """
        def __init__(self, ninputs, ksize, noutputs, normalize=False,
                     normalization_type="batch", stride=1, padding=0,
                     activation="relu", weight_norm=True):
            super(ModifiedConvChain._ConvBNRelu, self).__init__()

            if activation == "relu":
                act_fn = nn.ReLU
            elif activation == "leaky_relu":
                act_fn = nn.LeakyReLU
            elif activation == "tanh":
                act_fn = nn.Tanh
            elif activation == "elu":
                act_fn = nn.ELU
            else:
                raise ValueError("activation should be one of: "
                                 "relu, leaky_relu, tanh, elu")

            if normalize:
                print("nrm", normalization_type)
                conv = nn.Conv2d(ninputs, noutputs, ksize,
                                 stride=stride, padding=padding, bias=False)
                if normalization_type == "batch":
                    nrm = nn.BatchNorm2d(noutputs)
                elif normalization_type == "instance":
                    nrm = nn.InstanceNorm2D(noutputs)
                else:
                    raise ValueError(
                        "Unkown normalization type {}".format(
                            normalization_type))
                nrm.bias.data.zero_()
                nrm.weight.data.fill_(1.0)
                self.layer = nn.Sequential(conv, nrm, act_fn())
            else:
                conv = nn.Conv2d(ninputs, noutputs, ksize,
                                 stride=stride, padding=padding)
                if weight_norm:
                    conv = nn.utils.weight_norm(conv)
                conv.bias.data.zero_()
                self.layer = nn.Sequential(conv, act_fn())

            if activation == "elu":
                nn.init.xavier_uniform_(
                    conv.weight.data, nn.init.calculate_gain("relu"))
            else:
                nn.init.xavier_uniform_(
                    conv.weight.data, nn.init.calculate_gain(activation))

        def forward(self, x):
            out = self.layer(x)
            return out