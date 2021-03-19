import torch

from . import tools


class UnitTCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(UnitTCN, self).__init__()

        self.bn = torch.nn.BatchNorm2d(
            num_features=out_channels,
            eps=1e-5,
            momentum=0.1,
            affine=True,
            track_running_stats=True,
        )

        self.conv = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(kernel_size, 1),
            stride=(stride, 1),
            padding=(int((kernel_size - 1) / 2), 0),
            dilation=1,
            groups=1,
            bias=True,
            padding_mode="zeros",  # 'zeros', 'reflect', 'replicate', 'circular'
        )

        tools.init_bn(self.bn, 1)
        tools.init_conv(self.conv)

    def forward(self, x):
        return self.bn(self.conv(x))
