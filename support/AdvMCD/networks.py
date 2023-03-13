import torch
import torch.nn as nn
from .modules import ResBlock_CFM, CFMLayer

class Generator(nn.Module):
    """
    Denoising model(i.e., Generator) based on direct prediction (Xu et al. 2019)
    https://github.com/mcdenoising/AdvMCDenoise/blob/master/codes/models/arch/generator_cfm.py
    """
    def __init__(self, feat_ch=7, res_ch=64, cond_ch=128, n_resblock=16):
        super(Generator, self).__init__()
        self.conv0 = nn.Conv2d(3, res_ch, 3, 1, 1)

        CFM_branch = []
        for i in range(n_resblock):
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
            nn.Conv2d(feat_ch, cond_ch, 3, 1, 1),
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