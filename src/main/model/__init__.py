from .stream_spatial import *
from .stream_temporal import *
from functools import *
from torch import nn


class TKNet(nn.Module):
    def __init__(
        self,
        stream=[0,1],
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

        self.stream_indices = stream
        
        self.streams = nn.ModuleList([StreamSpatialGCN(input_size=input_size, cls_graph=cls_graph),
                        StreamTemporalGCN(input_size=input_size, cls_graph=cls_graph)])

        num_stream_units=[64, 300]

        num_concat_units= sum(num_stream_units[i] for i in stream)

        self.fc1 = nn.Linear(num_concat_units, num_concat_units//2)
        
        self.fc2 = nn.Linear(num_concat_units//2, num_class)

    def forward(self, x):

        output_streams =  tuple(map(lambda i: self.streams[i](x), self.stream_indices))

        output_concat = torch.cat(output_streams, dim=1)
        
        return self.fc2(self.fc1(output_concat))
