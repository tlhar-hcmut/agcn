import math

import numpy as np
import torch
from src.main.graph import NtuGraph
from torch import autograd, nn
from torch.nn import *


class StreamSpatialGCN(Module):
    def __init__(
        self, num_class=12, num_point=25, num_person=2, in_channels=3, pre_train=True
    ):
        super(StreamSpatialGCN, self).__init__()

        self.graph = NtuGraph()

        A = self.graph.A
        self.data_bn = BatchNorm1d(num_person * in_channels * num_point)

        self.l1 = Unit(3, 64, A, residual=False)
        self.l2 = Unit(64, 64, A)
        self.l3 = Unit(64, 64, A)
        self.l4 = Unit(64, 64, A)
        self.l5 = Unit(64, 128, A, stride=2)
        self.l6 = Unit(128, 128, A)
        self.l7 = Unit(128, 128, A)
        self.l8 = Unit(128, 256, A, stride=2)
        self.l9 = Unit(256, 256, A)
        self.l10 = Unit(256, 256, A)

        self.fc = Linear(256, num_class)
        init.normal_(self.fc.weight, 0, math.sqrt(2.0 / num_class))
        init_bn(self.data_bn, 1)

        if pre_train:
            weight = torch.load("weight/stream-spatial.pt")
            weight.pop("fc.bias")
            weight.pop("fc.weight")
            self.load_state_dict(weight, strict=False)

    def forward(self, x):
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = (
            x.view(N, M, V, C, T)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
            .view(N * M, C, T, V)
        )

        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)

        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)

        return self.fc(x)


class Unit(Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True):
        super(Unit, self).__init__()
        self.gcn1 = Gcn(in_channels, out_channels, A)
        self.tcn1 = Tcn(out_channels, out_channels, stride=stride)
        self.relu = ReLU()
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = Tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        return self.relu(self.tcn1(self.gcn1(x)) + self.residual(x))


class Tcn(Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(Tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
        )
        self.bn = BatchNorm2d(out_channels)
        init_conv(self.conv)
        init_bn(self.bn, 1)

    def forward(self, x):
        return self.bn(self.conv(x))


class Gcn(Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, num_subset=3):
        super(Gcn, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.num_subset = num_subset

        self.conv_a = ModuleList()
        self.conv_b = ModuleList()
        self.conv_d = ModuleList()

        mat_adj = torch.from_numpy(A.astype(np.float32))
        self.PA = nn.Parameter(mat_adj)
        self.A = autograd.Variable(mat_adj)

        init.constant_(self.PA, 1e-6)

        for i in range(self.num_subset):
            self.conv_a.append(Conv2d(in_channels, inter_channels, 1))
            self.conv_b.append(Conv2d(in_channels, inter_channels, 1))
            self.conv_d.append(Conv2d(in_channels, out_channels, 1))

        if in_channels != out_channels:
            self.down = Sequential(
                Conv2d(in_channels, out_channels, 1), BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn = BatchNorm2d(out_channels)
        self.soft = Softmax(-2)
        self.relu = ReLU()

        for m in self.modules():
            if isinstance(m, Conv2d):
                init_conv(m)
            elif isinstance(m, BatchNorm2d):
                init_bn(m, 1)
        init_bn(self.bn, 1e-6)
        for i in range(self.num_subset):
            init_conv_branch(self.conv_d[i], self.num_subset)

    def forward(self, x):
        N, C, T, V = x.size()
        A = self.A.cuda(x.get_device())
        A = A + self.PA

        y = None
        for i in range(self.num_subset):
            A1 = (
                self.conv_a[i](x)
                .permute(0, 3, 1, 2)
                .contiguous()
                .view(N, V, self.inter_c * T)
            )
            A2 = self.conv_b[i](x).view(N, self.inter_c * T, V)
            A1 = self.soft(torch.matmul(A1, A2) / A1.size(-1))  # N V V
            A1 = A1 + A[i]
            A2 = x.view(N, C * T, V)
            z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))
            y = z + y if y is not None else z

        y = self.bn(y)
        y += self.down(x)
        return self.relu(y)


def init_conv_branch(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    init.normal_(weight, 0, math.sqrt(2.0 / (n * k1 * k2 * branches)))
    init.constant_(conv.bias, 0)


def init_conv(conv):
    init.kaiming_normal_(conv.weight, mode="fan_out")
    init.constant_(conv.bias, 0)


def init_bn(bn, scale):
    init.constant_(bn.weight, scale)
    init.constant_(bn.bias, 0)
