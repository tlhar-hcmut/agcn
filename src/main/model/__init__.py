from functools import *

import src.main.config.cfg_train as cfg_train
import torch.nn.functional as F
from torch import nn

from .stream_spatial import *
from .stream_temporal import *


class TKNet(nn.Module):
    def __init__(
        self,
        name="",
        stream=[0, 1],
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

        self.input_size = input_size
        self.name = name

        self.stream_indices = stream

        self.streams = nn.ModuleList(
            [
                StreamSpatialGCN(),
                StreamTemporalGCN(
                    input_size=input_size,
                    cls_graph=cls_graph,
                    dropout=cfg_train.dropout,
                    num_head=cfg_train.num_head,
                    num_block=cfg_train.num_block,
                    len_feature_new=cfg_train.len_feature_new,
                ),
            ]
        )

        num_stream_units = [64, 300]

        num_concat_units = sum(num_stream_units[i] for i in stream)

        self.fc1 = nn.Linear(num_concat_units, 64)

        self.fc2 = nn.Linear(64, num_class)

    def forward(self, x):

        output_streams = tuple(map(lambda i: self.streams[i](x), self.stream_indices))

        output_concat = torch.cat(output_streams, dim=1)

        return self.fc2(F.relu(self.fc1(output_concat)))


class KhoiDDNet(nn.Module):
    def __init__(self, name="", input_size=(3, 300, 25, 2), num_class=12):
        super(KhoiDDNet, self).__init__()
        self.name = name
        self.input_size = input_size

        self.stream_spatial = StreamSpatialGCN()
        self.stream_temporal = StreamTemporalGCN(
            input_size=(256, 300, 25, 2),
            dropout=cfg_train.dropout,
            num_head=cfg_train.num_head,
            num_block=cfg_train.num_block,
            len_feature_new=cfg_train.len_feature_new,
        )
        self.fc = nn.Linear(32, num_class)

    def forward(self, x):
        return self.fc(self.stream_temporal(self.stream_spatial(x)))
