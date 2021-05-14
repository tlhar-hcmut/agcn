from .stream_spatial import *
from .stream_temporal import *
from functools import *
from torch import nn


class TKNet(nn.Module):
    def __init__(
        self,
        stream=[0, 1],
        num_class=60,
        cls_graph=None,
        graph_args=dict(),
        **kargs
    ):
        super(TKNet, self).__init__()

        self.stream = StreamSpatialGCN()
        self.fc = nn.Linear(256, num_class)

    def forward(self, x):
        return self.fc(self.stream(x))
