import math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from importer import import_class
from initor import *


class UnitTcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(UnitTcn, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(kernel_size, 1),
            stride=(stride, 1),
            padding=(int((kernel_size - 1) / 2), 0),
            dilation=1,
            groups=1,
            bias=True,
            padding_mode='zeros',  # 'zeros', 'reflect', 'replicate', 'circular'
        )

        self.bn = nn.BatchNorm2d(
            num_features=out_channels,
            eps=1e-5,
            momentum=0.1,
            affine=True,
            track_running_stats=True,
        )

        nn.init.constant_(bn.bias, 0)
        nn.init.constant_(bn.weight, 1)

        nn.init.constant_(conv.bias, 0)
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')

    def forward(self, x):
        return self.bn(self.conv(x))


class UnitGcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, num_subset=3):
        super(UnitGcn, self).__init__()
        self.inter_c = out_channels // coff_embedding
        self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        nn.init.constant_(self.PA, 1e-6)
        self.A = Variable(
            torch.from_numpy(A.astype(np.float32)),
            requires_grad=False)
        self.num_subset = num_subset

        self.conv_a = nn.ModuleList()
        self.conv_b = nn.ModuleList()
        self.conv_d = nn.ModuleList()
        for i in range(self.num_subset):
            self.conv_a.append(nn.Conv2d(in_channels, self.inter_c, 1))
            self.conv_b.append(nn.Conv2d(in_channels, self.inter_c, 1))
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))

        if in_channels != out_channels:
            self.conv_res = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.conv_res = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_conv(m)
            elif isinstance(m, nn.BatchNorm2d):
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
            A1 = self.conv_a[i](x)\
                .permute(0, 3, 1, 2)\
                .contiguous().view(N, V, self.inter_c * T)
            A2 = self.conv_b[i](x).view(N, self.inter_c * T, V)
            A1 = self.soft(torch.matmul(A1, A2) / A1.size(-1))  # N V V
            A1 = A1 + A[i]
            A2 = x.view(N, C * T, V)
            z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))
            y = z + y if y is not None else z

        y = self.bn(y)
        y += self.conv_res(x)
        return self.relu(y)


class UnitTcnGcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True):
        super(UnitTcnGcn, self).__init__()
        self.gcn1 = UnitGcn(in_channels, out_channels, A)
        self.tcn1 = UnitTcn(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU()
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = UnitTcn(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride
            )

    def forward(self, x):
        x = self.tcn1(self.gcn1(x)) + self.residual(x)
        return self.relu(x)


class Model(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3):
        super(Model, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        self.l1 = UnitTcnGcn(3, 64, A, residual=False)
        self.l2 = UnitTcnGcn(64, 64, A)
        self.l3 = UnitTcnGcn(64, 64, A)
        self.l4 = UnitTcnGcn(64, 64, A)
        self.l5 = UnitTcnGcn(64, 128, A, stride=2)
        self.l6 = UnitTcnGcn(128, 128, A)
        self.l7 = UnitTcnGcn(128, 128, A)
        self.l8 = UnitTcnGcn(128, 256, A, stride=2)
        self.l9 = UnitTcnGcn(256, 256, A)
        self.l10 = UnitTcnGcn(256, 256, A)

        self.fc = nn.Linear(256, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        init_bn(self.data_bn, 1)

    def forward(self, x):
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(
            0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

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


if __name__ == "__main__":
    model = Model(graph='graph.ntu_rgb_d.Graph')  # default
    print(model)
