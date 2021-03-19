import math

import torch

from . import UnitTGCN, tools


class UnitAGCN(torch.nn.Module):
    def __init__(
        self,
        num_class=60,
        num_point=25,
        num_person=2,
        cls_graph=None,
        graph_args=dict(),
        in_channels=3,
    ):
        super(UnitAGCN, self).__init__()

        if cls_graph is None:
            raise ValueError()
        else:
            self.graph = cls_graph(**graph_args)

        A = self.graph.A
        self.data_bn = torch.nn.BatchNorm1d(num_person * in_channels * num_point)

        self.l1 = UnitTGCN(3, 8, A, residual=False)
        self.l2 = UnitTGCN(8, 8, A)
        self.l3 = UnitTGCN(8, 8, A)
        self.l4 = UnitTGCN(8, 8, A)
        self.l5 = UnitTGCN(8, 16, A, stride=2)
        self.l6 = UnitTGCN(16, 16, A)
        self.l7 = UnitTGCN(16, 16, A)
        self.l8 = UnitTGCN(16, 32, A, stride=2)
        self.l9 = UnitTGCN(32, 32, A)
        self.l10 = UnitTGCN(32, 32, A)

        self.fc = torch.nn.Linear(32, num_class)

        torch.nn.init.normal_(self.fc.weight, 0, math.sqrt(2.0 / num_class))
        tools.init_bn(self.data_bn, 1)

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


if __name__ == "__main__":
    model = UnitAGCN(graph="graph.ntu_rgb_d.Graph")  # default
    print(model)