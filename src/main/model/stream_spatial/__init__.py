import math

import torch

from . import util
from .agcn import UnitAGCN
from .tagcn import UnitTAGCN
from .tcn import UnitTCN


class StreamSpatialGCN(torch.nn.Module):
    def __init__(
        self,
        num_class=60,
        num_point=25,
        num_person=2,
        cls_graph=None,
        graph_args=dict(),
        in_channels=3,
    ):
        super(StreamSpatialGCN, self).__init__()

        if cls_graph is None:
            raise ValueError()
        else:
            self.graph = cls_graph(**graph_args)

        A = self.graph.A
        self.data_bn = torch.nn.BatchNorm1d(num_person * in_channels * num_point)

        self.l1 = UnitTAGCN(3, 16, A, residual=False)
        self.l2 = UnitTAGCN(16, 16, A)
        self.l3 = UnitTAGCN(16, 16, A)
        self.l4 = UnitTAGCN(16, 16, A)
        self.l5 = UnitTAGCN(16, 32, A, stride=2)
        self.l6 = UnitTAGCN(32, 32, A)
        self.l7 = UnitTAGCN(32, 32, A)
        self.l8 = UnitTAGCN(32, 64, A, stride=2)
        self.l9 = UnitTAGCN(64, 64, A)
        self.l10 = UnitTAGCN(64, 64, A)

        self.fc = torch.nn.Linear(64, num_class)

        torch.nn.init.normal_(self.fc.weight, 0, math.sqrt(2.0 / num_class))
        util.init_bn(self.data_bn, 1)

    def forward(self, x):
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = (
            x.contiguous()
            .view(N, M, V, C, T)
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
        x = x.contiguous().view(N, M, c_new, -1)
        x = x.mean(3).mean(1)
        # x = self.fc(x)

        return x
