import torch
import torch.nn as nn
import torch.nn.functional as F


class _DomainSpecificBatchNorm(nn.Module):
    _version = 2

    def __init__(self, num_features, num_classes, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(_DomainSpecificBatchNorm, self).__init__()
        #         self.bns = nn.ModuleList([nn.modules.batchnorm._BatchNorm(num_features, eps, momentum, affine, track_running_stats) for _ in range(num_classes)])
        self.bns = nn.ModuleList(
            [nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats) for _ in range(num_classes)])

    def reset_running_stats(self):
        for bn in self.bns:
            bn.reset_running_stats()

    def reset_parameters(self):
        for bn in self.bns:
            bn.reset_parameters()

    def _check_input_dim(self, input):
        raise NotImplementedError

    def forward(self, x, domain_label):
        self._check_input_dim(x)
        bn = self.bns[domain_label]
        return bn(x)


class DomainSpecificBatchNorm2d(_DomainSpecificBatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(input.dim()))


class SingleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        # self.conv = nn.Sequential(
        #     nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(mid_channels),
        #     nn.ReLU(inplace=True),
        # )
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.bn1 = DomainSpecificBatchNorm2d(mid_channels, 2)
        self.act = nn.LeakyReLU(inplace=True)

    def forward(self, x, domain):
        return self.act(self.bn1(self.conv1(x), domain))


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, activation="relu"):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        
        if activation == "linear":
            self.act_fn = nn.Identity()
        elif activation == "relu":
            self.act_fn =  nn.ReLU(inplace=True)
        elif activation == "leaky_relu":
            self.act_fn = nn.LeakyReLU(inplace=True)
        else:
            raise ValueError("Unknon output type '{}'".format(activation))

        # self.double_conv = nn.Sequential(
        #     nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(mid_channels),
        #     act_fn, #nn.ReLU(inplace=True),
        #     nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(out_channels),
        #     act_fn # nn.ReLU(inplace=True)
        # )
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.bn1 = DomainSpecificBatchNorm2d(mid_channels, 2)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = DomainSpecificBatchNorm2d(out_channels, 2)
        # act_fn # nn.ReLU(inplace=True)

    def forward(self, x, domain):
        x = self.act_fn(self.bn1(self.conv1(x), domain))
        x = self.act_fn(self.bn2(self.conv2(x), domain))
        return x


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, activation="relu"):
        super().__init__()
        # self.maxpool_conv = nn.Sequential(
        #     nn.MaxPool2d(2),
        #     # nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2),
        #     DoubleConv(in_channels, out_channels, activation=activation)
        # )
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels, activation=activation)

    def forward(self, x, domain):
        return self.conv(self.pool(x), domain)


class StridedDown(nn.Module):
    """Downscaling with strided conv"""
    def __init__(self, in_channels, out_channels, activation="relu"):
        super().__init__()
        # self.strided_conv = nn.Sequential(
        #     nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2),         # TODO fix!!!
        #     DoubleConv(in_channels, out_channels, activation=activation)
        # )
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(in_channels, out_channels, activation=activation)

    def forward(self, x, domain):
        return self.conv2(self.conv1(x), domain)

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

    def forward(self, x1, x2, domain):
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
        return self.conv(x, domain)


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


class DSBNUNet(nn.Module):
    def __init__(self, n_channels, n_classes, hidden=64, bilinear=True, strided_down=False, activation="leaky_relu"):
        super(DSBNUNet, self).__init__()
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

    def forward(self, x, domain):
        x1 = self.inc(x, domain)
        # print('inc x1', x1.shape)
        x2 = self.down1(x1, domain)
        # print('down x2', x2.shape)
        x3 = self.down2(x2, domain)
        # print('down x3', x3.shape)
        x4 = self.down3(x3, domain)
        # print('down x4', x4.shape)
        x5 = self.down4(x4, domain)
        # print('down x5', x5.shape)
        x = self.up1(x5, x4, domain)
        x = self.up2(x, x3, domain)
        # print('up x3', x.shape)
        x = self.up3(x, x2, domain)
        # print('up x2', x.shape)
        x = self.up4(x, x1, domain)
        # print('up x1', x.shape)
        x = self.outc(x)
        return x, x5