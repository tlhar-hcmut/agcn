from .stream_spatial import *
from .stream_temporal import *
from .stream_temporal_test import *
from .stream_spatial_test import *
from functools import *
from torch import nn
import torch.nn.functional as F

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

        self.stream_indices = stream
        self.streams = nn.ModuleList([StreamTemporalGCN(**kargs),
                                      stream_temporal.StreamTemporalGCN(**kargs),
                                      stream_temporal_test.StreamTemporalGCN(**kargs)])

        num_stream_units = [64, 300, 300]

        num_concat_units = sum(num_stream_units[i] for i in stream)

        self.fc1 = nn.Linear(num_concat_units, 128)       

        self.ln1 =nn.LayerNorm(normalized_shape=(128)) 

        self.fc2 = nn.Linear(128, 64)

        self.ln2 =nn.LayerNorm(normalized_shape=(64)) 

        self.ln3 =nn.LayerNorm(normalized_shape=(num_class)) 

        self.fc3 = nn.Linear(64, num_class)





    def forward(self, x):

        output_streams = tuple(map(lambda i: self.streams[i](x), self.stream_indices))

        output = torch.cat(output_streams, dim=1)

        output = self.fc1(output)
        
        output =  self.ln1(output)

        output =  F.relu(output)

        output = self.fc2(output)
        
        output =  self.ln2(output)

        output =  F.relu(output)

        output = self.fc3(output)
        
        output =  self.ln3(output)
        
        return output


class SquentialNet(nn.Module):
    def __init__(
        self,
        stream=[0, 1],
        num_class=60,
        cls_graph=None,
        graph_args=dict(),
        **kargs
    ):
        super(SquentialNet, self).__init__()

        self.net = nn.Sequential(
            stream_spatial_test.StreamSpatialGCN(**kargs),
            stream_temporal_test.StreamTemporalGCN(**kargs))

        self.fc1 = nn.Linear(300, 128)       

        self.ln1 =nn.LayerNorm(normalized_shape=(128)) 

        self.fc2 = nn.Linear(128, 64)

        self.ln2 =nn.LayerNorm(normalized_shape=(64)) 

        self.ln3 =nn.LayerNorm(normalized_shape=(num_class)) 

        self.fc3 = nn.Linear(64, num_class)





    def forward(self, x):

        output = self.net(x)

        output = self.fc1(output)
        
        output =  self.ln1(output)

        output =  F.relu(output)

        output = self.fc2(output)
        
        output =  self.ln2(output)

        output =  F.relu(output)

        output = self.fc3(output)
        
        output =  self.ln3(output)
        
        return output

