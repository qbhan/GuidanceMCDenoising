import torch
import torch.nn as nn
import torch.nn.functional as F
from support.utils import crop_like

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


class Generator(nn.Module):
    """
    Denoising model(i.e., Generator) based on direct prediction (Xu et al. 2019)
    https://github.com/mcdenoising/AdvMCDenoise/blob/master/codes/models/arch/generator_cfm.py
    """
    def __init__(self, res_ch=64, cond_ch=128):
        super(Generator, self).__init__()
        self.conv0 = nn.Conv2d(3, res_ch, 3, 1, 1)

        CFM_branch = []
        for i in range(16):
            CFM_branch.append(ResBlock_CFM())
        CFM_branch.append(CFMLayer())
        CFM_branch.append(nn.Conv2d(res_ch, res_ch, 3, 1, 1))
        self.CFM_branch = nn.Sequential(*CFM_branch)

        self.Final_stage = nn.Sequential(
            nn.Conv2d(res_ch, res_ch, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(res_ch, 3, 3, 1, 1),
            nn.ReLU(True)
        )

        self.Condition_process = nn.Sequential(
            nn.Conv2d(7, cond_ch, 3, 1, 1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(cond_ch, cond_ch, 1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(cond_ch, cond_ch, 1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(cond_ch, cond_ch, 1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(cond_ch, 32, 1)
        )


    def forward(self, x):
        # x[0]: img; x[1]: seg
        cond = self.Condition_process(x[1]) #CHANGED
        
        fea = self.conv0(x[0])
        res = self.CFM_branch((fea, cond))
        fea = fea + res
        out = self.Final_stage(fea)
        return out



class Discriminator(nn.Module):
    '''
    Discriminator based on VGG with input size of 128*128 (Xu et al. 2019)
    https://github.com/mcdenoising/AdvMCDenoise/blob/master/codes/models/arch/discriminator_dispatcher.py
    '''
    def __init__(self, in_nc=3, base_nf=64, norm_type='batch', act_type='leakyrelu', mode='CNA'):
        super(Discriminator, self).__init__()
        # features
        # hxw, c
        # 128, 64
     
        def conv_block(in_nc, out_nc, kernel_size, stride=1, dilation=1, groups=1, bias=True, \
               pad_type='zero', norm_type=None, act_type='relu', mode='CNA'):
            '''
            Conv layer with padding, normalization, activation
            mode: CNA --> Conv -> Norm -> Act
                NAC --> Norm -> Act --> Conv (Identity Mappings in Deep Residual Networks, ECCV16)
            '''
            assert mode in ['CNA', 'NAC', 'CNAC'], 'Wong conv mode [{:s}]'.format(mode)
            # padding = get_valid_padding(kernel_size, dilation)
            # p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
            # padding = padding if pad_type == 'zero' else 0
            padding = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2

            c = nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding, \
                    dilation=dilation, bias=bias, groups=groups)
            a = nn.LeakyReLU(0.2, inplace=True)
            n = nn.BatchNorm2d(out_nc, affine=True)
            # return sequential(p, c, n, a)
            return nn.Sequential(c, n, a)


        conv0 = conv_block(in_nc, base_nf, kernel_size=3, norm_type=None, act_type=act_type, \
            mode=mode)
        conv1 = conv_block(base_nf, base_nf, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 64, 64
        conv2 = conv_block(base_nf, base_nf*2, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv3 = conv_block(base_nf*2, base_nf*2, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 32, 128
        conv4 = conv_block(base_nf*2, base_nf*4, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv5 = conv_block(base_nf*4, base_nf*4, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 16, 256
        conv6 = conv_block(base_nf*4, base_nf*8, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv7 = conv_block(base_nf*8, base_nf*8, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 8, 512
        conv8 = conv_block(base_nf*8, base_nf*8, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv9 = conv_block(base_nf*8, base_nf*8, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 4, 512
        self.features = nn.Sequential(conv0, conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8,\
            conv9)

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(512 * 4 * 4, 100), nn.LeakyReLU(0.2, True), nn.Linear(100, 1))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


'''
Modules for Making UNet
'''

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

    def __init__(self, in_channels, out_channels, mid_channels=None, activation="leaky_relu"):
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
            # nn.BatchNorm2d(mid_channels),
            act_fn, #nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(out_channels),
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
            # nn.LeakyReLU(inplace=True)
            DoubleConv(in_channels, out_channels, activation=activation)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class StridedDown(nn.Module):
    """Downscaling with strided conv"""
    def __init__(self, in_channels, out_channels, activation="relu"):
        super().__init__()
        self.strided_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=2, stride=2),         # TODO fix!!!
            DoubleConv(in_channels, out_channels, activation=activation)
        )

    def forward(self, x):
        return self.strided_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=False, activation="relu"):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, activation=activation)
            # self.conv = nn.Sequential(
            #     nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2),
            #     nn.LeakyReLU(inplace=True)
            # )
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, activation=activation)
            # self.conv = nn.Sequential(
            #     nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2),
            #     nn.LeakyReLU(inplace=True)
            # )

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
    def __init__(self, in_channels, out_channels, activation=torch.NoneType):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        if activation == 'sigmoid':
            self.act = nn.Sigmoid()
        else:
            self.act = nn.Identity()
    def forward(self, x):
        return self.act(self.conv(x))


class SimpleUNet(nn.Module):
    def __init__(self, n_channels, n_classes, hidden=64, bilinear=True, strided_down=True, activation="relu"):
        super(SimpleUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, hidden, activation=activation)
        
        if strided_down:
            down = StridedDown
        else:
            down = Down
        
        self.down1 = down(hidden, hidden*2, activation=activation)
        self.down2 = down(hidden*2, hidden*4, activation=activation)
        # self.down3 = down(hidden*2, hidden*4, activation=activation)
        # self.down4 = down(hidden*4, hidden*4, activation=activation)

        # self.up1 = Up(hidden*8, hidden*4, bilinear, activation=activation)
        # self.up2 = Up(hi dden*6, hidden*2, bilinear, activation=activation)
        self.up3 = Up(hidden*6, hidden*2, bilinear, activation=activation)
        self.up4 = Up(hidden*3, hidden, bilinear, activation=activation)
        self.outc = OutConv(hidden, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        # print('inc x1', x1.shape)
        x2 = self.down1(x1)
        # print('down x2', x2.shape)
        x3 = self.down2(x2)
        # print('down x3', x3.shape)
        # x4 = self.down3(x3)
        # print('down x4', x4.shape)
        # x5 = self.down4(x4)
        # print('down x5', x5.shape)
        # x = self.up1(x5, x4)
        # x = self.up2(x, x3)
        # print('up x3', x.shape)
        x = self.up3(x3, x2)
        # print('up x2', x.shape)
        x = self.up4(x, x1)
        # print('up x1', x.shape)
        x = self.outc(x)
        return x
    
    
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
