import torch
from torch import nn

from . import util
from .sgcn import UnitSpatialGcn


class TKNet(nn.Module):
    def __init__(
        self,
        input_size=(3, 300, 25, 2),
        num_class=60,
        cls_graph=None,
        graph_args=dict(),
    ):
        super(TKNet, self).__init__()

        if cls_graph is None:
            raise ValueError()
        else:
            self.graph = cls_graph(**graph_args)

        self.stream_spatial = StreamSpatialGCN(
            input_size=input_size, cls_graph=cls_graph
        )
        self.fc = nn.Linear(32, num_class)

    def forward(self, x):
        return self.fc(self.stream_spatial(x))


class StreamSpatialGCN(torch.nn.Module):
    def __init__(
        self, input_size, cls_graph=None, graph_args=dict(),
    ):
        super(StreamSpatialGCN, self).__init__()

        C, T, V, M = input_size
        num_person = M
        in_channels = C
        num_joint = V

        if cls_graph is None:
            raise ValueError()
        else:
            self.graph = cls_graph(**graph_args)

        A = self.graph.A
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_joint)

        self.l1 = UnitSpatialGcn(3, 8, A, residual=False)
        self.l2 = UnitSpatialGcn(8, 8, A)
        self.l3 = UnitSpatialGcn(8, 8, A)
        self.l4 = UnitSpatialGcn(8, 8, A)
        self.l5 = UnitSpatialGcn(8, 16, A, stride=2)
        self.l6 = UnitSpatialGcn(16, 16, A)
        self.l7 = UnitSpatialGcn(16, 16, A)
        self.l8 = UnitSpatialGcn(16, 32, A, stride=2)
        self.l9 = UnitSpatialGcn(32, 32, A)
        self.l10 = UnitSpatialGcn(32, 32, A)

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

        return x
