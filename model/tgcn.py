import torch
import numpy as np
from model import tools, UnitTCN, UnitGCN


class UnitTGCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, mat_adj, stride=1, residual=True):
        super(UnitTGCN, self).__init__()
        self.gcn1 = UnitGCN(in_channels, out_channels, mat_adj)
        self.tcn1 = UnitTCN(out_channels, out_channels, stride=stride)
        self.relu = torch.nn.ReLU()
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = UnitTCN(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride
            )

    def forward(self, x):
        x = self.tcn1(self.gcn1(x)) + self.residual(x)
        return self.relu(x)


if __name__ == "__main__":
    model = UnitTGCN(3, 64, np.ones((25, 25)))
    print(model)