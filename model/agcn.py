import math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from importer import import_class
from model import tools


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

        tools.init_bn(self.bn, 1)
        tools.init_conv(self.conv)

    def forward(self, x):
        return self.bn(self.conv(x))


class UnitGcn(nn.Module):
    def __init__(self, in_channels, out_channels, mat_adj, coff_embedding=4, num_subset=3):
        super(UnitGcn, self).__init__()
        # Init constant
        self.num_subset = num_subset
        self.inter_channels = out_channels // coff_embedding
        # Init cnn layer
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.soft = nn.Softmax(-2)
        # Init gcn layler
        self.mat_adj = Variable(
            data=torch.from_numpy(mat_adj.astype(np.float32)),
            requires_grad=False,
        )
        self.weight = nn.Parameter(
            data=torch.from_numpy(mat_adj.astype(np.float32)),
            requires_grad=True,
        )
        nn.init.constant_(self.weight, 1e-6)
        # Init embedding layer
        self.conv_a = nn.ModuleList()
        self.conv_b = nn.ModuleList()
        self.conv_d = nn.ModuleList()
        for i in range(self.num_subset):
            self.conv_a.append(nn.Conv2d(in_channels, self.inter_channels, 1))
            self.conv_b.append(nn.Conv2d(in_channels, self.inter_channels, 1))
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))

        if in_channels != out_channels:
            self.conv_res = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.conv_res = lambda x: x

        # Init weight
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                tools.init_conv(m)
            elif isinstance(m, nn.BatchNorm2d):
                tools.init_bn(m, 1)

        tools.init_bn(self.bn, 1e-6)

        for i in range(self.num_subset):
            tools.init_conv_branch(self.conv_d[i], self.num_subset)

    def forward(self, x):
        N, C, T, V = x.size()
        mat_adj = self.mat_adj.cuda(x.get_device())
        mat_adj = mat_adj + self.weight

        y = None
        for i in range(self.num_subset):
            mat_embed_1 = self.conv_a[i](x)\
                .permute(0, 3, 1, 2).contiguous()\
                .view(N, V, self.inter_channels * T) # N-C,T,V -> N-V,T,C -> N-V,TC

            mat_embed_2 = self.conv_b[i](x)\
                .view(N, self.inter_channels * T, V) # N-C,T,V -> N-CT,V

            mat_enhance = self.soft(torch.matmul(mat_embed_1, mat_embed_2) / A1.size(-1))  # N-V,V
            
            mat_inpt = x.view(N, C * T, V) # N-CT,V
            mat_adapt = mat_adj[i] + mat_enhance # N-V,V

            z = self.conv_d[i](torch.matmul(mat_inpt, mat_adapt).view(N, C, T, V)) # N-C,T,V
            y = z + y if y is not None else z

        return self.relu(self.bn(y) + self.conv_res(x))


class UnitTcnGcn(nn.Module):
    def __init__(self, in_channels, out_channels, mat_adj, stride=1, residual=True):
        super(UnitTcnGcn, self).__init__()
        self.gcn1 = UnitGcn(in_channels, out_channels, mat_adj)
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
        tools.init_bn(self.data_bn, 1)

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
