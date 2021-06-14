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


class SequentialNet(nn.Module):
    def __init__(
        self,
        stream=[0, 1],
        num_class=60,
        cls_graph=None,
        graph_args=dict(),
        **kargs
    ):
        super(SequentialNet, self).__init__()

        self.spatial_net = stream_spatial_test.StreamSpatialGCN(**kargs)

        self.temporal_net =  stream_temporal_test.StreamTemporalGCN(**kargs)

        self.fc1 = nn.Linear(300, 64)       

        self.ln1 =nn.LayerNorm(normalized_shape=(64)) 

        self.fc2 = nn.Linear(64, 64)

        self.ln2 =nn.LayerNorm(normalized_shape=(64)) 

        self.fc3 = nn.Linear(64, num_class)

    def forward(self, x):

        output = self.spatial_net(x)

        output = self.temporal_net(output)

        output = self.fc1(output)
        
        output =  self.ln1(output)

        output =  F.relu(output)

        output = self.fc2(output)
        
        output =  self.ln2(output)

        output =  F.relu(output)

        output = self.fc3(output)
        
        
        return output


class TemporalNet(nn.Module):
    def __init__(
        self,
        num_class=60,
        num_block =0,
        input_size=None,
        len_feature_new = [],
        **kargs
    ):
        super(TemporalNet, self).__init__()

        self.temporal_net =  stream_temporal_test.StreamTemporalGCN(num_block=num_block, len_feature_new = len_feature_new,**kargs)

        self.fc1 = nn.Linear(input_size[1], 64)       

        self.ln1 =nn.LayerNorm(normalized_shape=(64)) 

        self.fc2 = nn.Linear(64, 64)

        self.ln2 =nn.LayerNorm(normalized_shape=(64)) 

        self.fc3 = nn.Linear(64, num_class)

    def forward(self, x):

        output = self.temporal_net(x)

        output = self.fc1(output)
        
        output =  self.ln1(output)

        output =  F.relu(output)

        output = self.fc2(output)
        
        output =  self.ln2(output)

        output =  F.relu(output)

        output = self.fc3(output)
        
        
        return output

class TemporalNet_Sum(nn.Module):
    def __init__(
        self,
        num_class,
        input_size,
        **kargs
    ):
        super(TemporalNet_Sum, self).__init__()

        self.temporal_net =  stream_temporal_test.StreamTemporalGCN_Sum(**kargs)

        self.fc1 = nn.Linear(input_size[1], 64)       

        self.ln1 =nn.LayerNorm(normalized_shape=(64)) 

        self.fc2 = nn.Linear(64, 64)

        self.ln2 =nn.LayerNorm(normalized_shape=(64)) 

        self.fc3 = nn.Linear(64, num_class)

    def forward(self, x):

        output = self.temporal_net(x)

        output = self.fc1(output)
        
        output =  self.ln1(output)

        output =  F.relu(output)

        output = self.fc2(output)
        
        output =  self.ln2(output)

        output =  F.relu(output)

        output = self.fc3(output)
        
        
        return output

