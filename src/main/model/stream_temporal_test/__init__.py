from torch.nn.modules.normalization import LayerNorm
import src.main.config.cfg_train as cfg_train
import torch
import torch.nn.functional as F
from torch import nn
import  src.main.util  as util
from torch.nn import BatchNorm2d, Module, Conv2d
from .transformer import TransformerEncoder


class StreamTemporalGCN(torch.nn.Module):
    def __init__(
        self, num_head, num_block, dropout, len_feature_new,input_size_temporal=(3,300,25,2), **kargs
    ):
        super(StreamTemporalGCN, self).__init__()

        C, T, V, M = input_size_temporal
        channels = C
        num_frame = T

        

        #conv 1x1
        # self.vConv1 = ConvNorm(V, V, (T,C))
        # self.vConv2 = ConvNorm(V,5, (T,C))
        # self.vConv3 = ConvNorm(5,5, (T,C))
        # self.vConv4 = ConvNorm(5,5, (T,C))
        # self.vConv5 = ConvNorm(5,1, (T,C))

        self.ln0 = nn.LayerNorm(normalized_shape=(num_frame,channels))

        self.transformer = TransformerEncoder(
            input_size_transformer=(num_frame, channels),
            len_feature_new=len_feature_new,
            num_block=num_block,
            len_seq=num_frame,
            dropout=dropout,
            num_head=num_head,
        )

        # self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.linear1 = nn.Linear(len_feature_new[num_block - 1], 32)
        self.ln1 = nn.LayerNorm(normalized_shape=(300,32))

        self.linear2 = nn.Linear(32, 32)
        self.ln2 = nn.LayerNorm(normalized_shape=(300,32))

        self.linear3 = nn.Linear(32, 1)

    def forward(self, X):

        N_0, C_0, T_0, V_0, M_0 = X.size()

        X = X.permute(0, 4, 3, 2, 1).contiguous().view(N_0 * M_0, V_0 , T_0, C_0)
        
        # [-1, V_0 , T_0, C_0 ]  -> [-1 , T_0, C_0]
        # X = F.gelu(self.vConv1(X))
        # X = F.gelu(self.vConv2(X))
        # X = F.gelu(self.vConv3(X))
        # X = F.gelu(self.vConv4(X))
        # X = F.gelu(self.vConv5(X)).squeeze()

        X = X.contiguous().view(N_0 * M_0* V_0 , T_0, C_0)

        # [-1, 300, C_0]  -> [-1, 300, C_new]
        X = self.transformer(X)

        # [-1, 300, C_new] -> [-1, 300, 1] -> [-1, 300]
        X = F.gelu(self.ln1(self.linear1(X)))
        X = F.gelu(self.ln2(self.linear2(X)))
        X = self.linear3(X).squeeze()

        X = X.view(N_0*M_0, V_0, T_0).mean(1).squeeze()

        # [-1, 300] -> [-1, 2, 300] -> [-1, 1, 300]
        X = X.view(N_0, M_0, -1)
        X = X.mean(1)

        return X

class ConvNorm(Module):
    def __init__(self, in_channels, out_channels, input_zize, kernel_size=(1,1), stride=(1,1)):
        super(ConvNorm, self).__init__()
        pad = tuple((s - 1) // 2 for s in kernel_size)
        self.conv = Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=pad,
            stride=stride,
        )
        self.bn = LayerNorm(input_zize)
        util.init_conv(self.conv)
        util.init_bn(self.bn, 1)

    def forward(self, x):
        return self.bn(self.conv(x))
