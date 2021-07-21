from .stream_temporal import *
from .stream_spatial import *
from functools import *
from torch import nn, isnan
import torch.nn.functional as F

class ParallelNet(nn.Module):
    def __init__(
        self,
        num_class=60,
        num_joint=25,
        **kargs
    ):
        super(ParallelNet, self).__init__()

        self.streams = nn.ModuleList([
                                      stream_spatial.StreamSpatialGCN(**kargs),
                                      stream_temporal.StreamTemporalGCN(**kargs)])

        self.fc1= nn.Linear(8*num_joint, 1)    # 8 is output channel of Spatial

        self.spatial = stream_spatial.StreamSpatialGCN(**kargs)

        self.temporal = stream_temporal.StreamTemporalGCN(**kargs)

        num_concat_units = 300 + 300

        self.fc2 = nn.Linear(num_concat_units, 128)       

        self.ln1 =nn.LayerNorm(normalized_shape=(128)) 

        self.fc3 = nn.Linear(128, 64)

        self.ln2 =nn.LayerNorm(normalized_shape=(64)) 

        self.ln3 =nn.LayerNorm(normalized_shape=(num_class)) 

        self.fc4 = nn.Linear(64, num_class)



    def forward(self, x):

        

        #feed forward
        output1 = self.spatial(x)

        N_0, C_0, T_0, V_0, M_0 = output1.size()
        output1 = output1.permute(0, 4, 2, 1, 3).contiguous().view(N_0 * M_0, T_0, V_0*C_0)
        output1 = self.fc1(output1)
        
        
        output1 = output1.view(N_0, M_0, -1)
        output1 = output1.mean(1)



        output2 = self.temporal(x)        

        output = torch.cat([output1, output2], dim=1)

        output = self.fc2(output)
        
        output =  self.ln1(output)

        output =  F.relu(output)

        output = self.fc3(output)
        
        output =  self.ln2(output)

        output =  F.relu(output)

        output = self.fc4(output)
        
        output =  self.ln3(output)
        
        return output

class SequentialNet(nn.Module):
    def __init__(
        self,
        num_class=60,
        **kargs
    ):
        super(SequentialNet, self).__init__()

        self.spatial_net = stream_spatial.StreamSpatialGCN(**kargs)

        self.temporal_net =  stream_temporal.StreamTemporalGCN(**kargs)

        self.fc1 = nn.Linear(300, 128)       

        self.ln1 =nn.LayerNorm(normalized_shape=(128)) 

        self.fc2 = nn.Linear(128, 128)

        self.ln2 =nn.LayerNorm(normalized_shape=(128)) 

        self.fc3 = nn.Linear(128, 64)

        self.ln3 =nn.LayerNorm(normalized_shape=(64)) 

        self.fc4 = nn.Linear(64, 64)

        self.ln4 =nn.LayerNorm(normalized_shape=(64)) 

        self.fc5 = nn.Linear(64, num_class)

    def forward(self, x, is_output_att=False):

        output = self.spatial_net(x)

        output = self.temporal_net(output, is_output_att)
        if is_output_att==True:
            #output is tuple of attentions
            return output
    
        output = F.relu(self.ln1(self.fc1(output)))

        output = F.relu(self.ln2(self.fc2(output)))

        output = F.relu(self.ln3(self.fc3(output)))

        output = F.relu(self.ln4(self.fc4(output)))

        output = self.fc5(output)
        
        
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

        self.temporal_net =  stream_temporal.StreamTemporalGCN(num_block=num_block, len_feature_new = len_feature_new,**kargs)

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

        self.temporal_net =  stream_temporal.StreamTemporalGCN_Sum(**kargs)

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

class AuthorNet(nn.Module):
    def __init__(
        self,
        num_class=60,
        num_joint=25,
        **kargs
    ):
        super(AuthorNet, self).__init__()

        self.spatial = stream_spatial.StreamSpatialGCNAuthor(**kargs)
        
        self.fc1= nn.Linear(32*num_joint, 1)    # 8 is output channel of Spatial

        self.fc2 = nn.Linear(75, 64)       

        self.ln2 =nn.LayerNorm(normalized_shape=(64)) 

        self.fc3 = nn.Linear(64, 64)

        self.ln3 =nn.LayerNorm(normalized_shape=(64)) 
        
        self.fc4 = nn.Linear(64, num_class)

        self.ln4 =nn.LayerNorm(normalized_shape=(num_class)) 



    def forward(self, x):

        #feed forward
        output = self.spatial(x)

        N_0, C_0, T_0, V_0, M_0 = output.size()
        output = output.permute(0, 4, 2, 1, 3).contiguous().view(N_0 * M_0, T_0, V_0*C_0)
        output = self.fc1(output).squeeze(-1)
        
        output = output.view(N_0, M_0, -1)
        output = output.mean(1)

        output = self.fc2(output)
        
        output =  self.ln2(output)

        output =  F.relu(output)

        output = self.fc3(output)
        
        output =  self.ln3(output)

        output =  F.relu(output)

        output = self.fc4(output)
        
        output =  self.ln4(output)

        output =  F.relu(output)

        
        return output

