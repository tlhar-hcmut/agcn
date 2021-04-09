import math

import torch
from torch import nn

from src.main.model.transformer import TransformerUnit

from . import util
from .tgcn import UnitTGCN
from torch.nn import Softmax 
from src.main.model.agcn import UnitAGCN


class Net(torch.nn.Module):
    def __init__(
        self,
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
        self.conv1 = nn.Conv2d(
            in_channels=300,
            out_channels=128,
            # N-T, C, V: 1 for C, and 3 for V
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0), #equal padding
            dilation=1,
            groups=1,
            bias=True,
            padding_mode="zeros",  # 'zeros', 'reflect', 'replicate', 'circular'
        )#300, 3, 25 -> 128, 3, 25



        self.transformer1 = TransformerUnit(T=128)
        self.transformer2 = TransformerUnit(T=128)
        self.transformer3 = TransformerUnit(T=128)

        self.conv2 = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=(2, 75),
            stride=(2, 1),
            padding=(0, 0),  # conv -> 0
            dilation=1,
            groups=1,
            bias=True,
            padding_mode="zeros",  # 'zeros', 'reflect', 'replicate', 'circular'
        )  #N, 128, 75 -> N, 64

        
        self.fc1= nn.Linear(64,num_class)
       

    def forward(self, x):
        #tream old
        stream_agcn = self.agcn(x)

        #stream transformer
        N_0, C, T, V, M_0 = x.size()

        # -> N_0-T, C, V
        stream_transformer = x.permute(0, 4, 2, 1, 3).contiguous().view(N_0*M_0, T, C, V).to("cuda")
        
        #N, 300, 3, 25 -> N, 128, 3, 25
        stream_transformer = self.conv1(stream_transformer)

        N, T, C, V = stream_transformer.size()

        #N, 128, 3, 25  -> N, 128, 75
        stream_transformer = stream_transformer.contiguous().view(N, T, C * V)

        stream_transformer = self.transformer1(stream_transformer)
        stream_transformer = self.transformer2(stream_transformer)
        stream_transformer = self.transformer3(stream_transformer)

        #N, 128, 75 -> N, 64
        stream_transformer = stream_transformer.unsqueeze(1)
        stream_transformer = self.conv2(stream_transformer).squeeze()
        
        stream_transformer = stream_transformer.view(N_0, M_0, 64)
        stream_transformer = stream_transformer.mean(1)
        
        stream_transformer = self.fc1(stream_transformer)

        
        return stream_agcn + stream_transformer
