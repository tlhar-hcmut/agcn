from .stream_spatial import *
from .stream_temporal import *
from .stream_temporal_test import *
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

        self.fc1 = nn.Linear(num_concat_units, 50)

        self.fc2 = nn.Linear(50, num_class)

        # self.fc3 = nn.Linear(num_concat_units, 300)

        # self.fc4 = nn.Linear(300, 100)

        # self.fc5 = nn.Linear(100, 100)

        # self.fc6 = nn.Linear(100, num_class)

        self.ln1 =nn.LayerNorm(normalized_shape=(50)) 

        self.ln2 =nn.LayerNorm(normalized_shape=(num_class)) 



    def forward(self, x):

        output_streams = tuple(
            map(lambda i: self.streams[i](x), self.stream_indices))

        output = torch.cat(output_streams, dim=1)

        output = self.fc1(output)
        
        output =  self.ln1(output)

        output =  F.relu(output)

        output = self.fc2(output)

        output = self.ln2(output)


        # output = F.gelu(self.fc3(output))

        # output = F.gelu(self.fc4(output))

        # output = F.gelu(self.fc5(output))

        # output = F.gelu(self.fc6(output))


        #test
        predict_labels = torch.argmax(output, 1).to("cpu")
        # print(predict_labels)
        #end test
        
        return output