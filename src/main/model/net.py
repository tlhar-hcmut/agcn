import math

import torch
from torch import nn

from src.main.model.stream_temporal.transformer import TransformerEncoder

from . import util
from .tgcn import UnitTGCN
from torch.nn import Softmax 
from src.main.model.agcn import UnitAGCN


class Net(torch.nn.Module):
    def __init__(
        self,
        device, 
        input_size =(150, 75),
        num_class=60,
        cls_graph=None,
        graph_args=dict(),
    ):
        super(Net, self).__init__()

        if cls_graph is None:
            raise ValueError()
        else:
            self.graph = cls_graph(**graph_args)

        #stream old
        self.agcn = UnitAGCN(num_class=num_class, cls_graph=cls_graph)

        #stream transformer

        #input: N, C, T, V
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=3,
            kernel_size=(2, 1),
            stride=(2, 1),
            padding=(0, 0), #poolling and equal padding
            dilation=1,
            groups=1,
            bias=True,
            padding_mode="zeros",  # 'zeros', 'reflect', 'replicate', 'circular'
        )#3, 300, 25 -> 3, 150, 25


        #N, 300, 128
        self.transformer = TransformerEncoder(device, input_size =input_size , len_seq=150)

        self.conv2 = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=(2, 4),
            stride=(2, 4),
            padding=(0, 0),
            dilation=1,
            groups=1,
            bias=True,
            padding_mode="zeros",  # 'zeros', 'reflect', 'replicate', 'circular'
        ) 

        
        self.fc1= nn.Linear(75,12)
        self.fc2= nn.Linear(24,num_class)
       

    def forward(self, x):
        #tream old
        stream_agcn = self.agcn(x)

        #stream transformer
        N_0, C, T, V, M_0 = x.size()

        # -> N-T, C, V
        stream_transformer = x.permute(0, 4, 2, 1, 3).contiguous().view(N_0*M_0, T, C, V)

        #N T, C , V => N C, T, V
        stream_transformer = stream_transformer.permute(0, 2, 1, 3)
        
        #N, 3, 300, 25 ->N, 3, 150, 25
        stream_transformer = self.conv1(stream_transformer)
        
        # N C, T, V => N T, C , V
        stream_transformer = stream_transformer.permute(0, 2, 1, 3)
 
        N, T, C, V = stream_transformer.size()

        #N, 150, 3, 25  -> N, 150, 75
        stream_transformer = stream_transformer.contiguous().view(N, T, C * V)
        #N, 150, 75  -> N, 150, 128
        stream_transformer = self.transformer(stream_transformer)

        #N, 150, 128 -> N, 75, 32
        stream_transformer = stream_transformer.unsqueeze(1)
        stream_transformer = self.conv2(stream_transformer).squeeze()

        #N, 75, 32 -> N, 75
        stream_transformer = torch.mean(stream_transformer,dim=-1)

        stream_transformer = stream_transformer.view(N_0, M_0, 64)
        stream_transformer = stream_transformer.mean(1)
        
        stream_transformer = self.fc1(stream_transformer)

        fusion = torch.cat((stream_agcn, stream_transformer),1)
        result = self.fc2(fusion)
        
        return result
