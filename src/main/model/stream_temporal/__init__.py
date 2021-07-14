from torch.nn.modules.normalization import LayerNorm
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
        self.len_feature_new = len_feature_new

        C, T, V, M = input_size_temporal

        self.ln0 = nn.LayerNorm(normalized_shape=(T,C*V))

        self.transformer = TransformerEncoder(
            input_size_transformer=(T, C*V),
            len_feature_new=len_feature_new,
            num_block=num_block,
            len_seq=T,
            dropout=dropout,
            num_head=num_head,
        )

        self.fc0 = nn.Linear(len_feature_new[num_block-1], 1, bias=True)


    def forward(self, X):

        N_0, C_0, T_0, V_0, M_0 = X.size()

        X = X.permute(0, 4, 2, 1, 3).contiguous().view(N_0 * M_0, T_0, V_0*C_0)

        Y = torch.sum(X,dim=-1)
        mask = torch.ones_like(Y, dtype=torch.int8)
        mask[Y==0]=0
        mask = mask.bool().to(X.get_device())

        X= self.ln0(X)
        
        # [-1, 300, C_0]  -> [-1, 300, C_new]
        X = self.transformer(X, mask)

        # [-1, 300, C_new] -> [-1, 300]
        X = self.fc0(X).squeeze(-1)
        X = X*mask
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
