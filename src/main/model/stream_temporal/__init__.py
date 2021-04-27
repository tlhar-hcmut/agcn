import torch
from torch import nn

from .add_norm import AddNorm
from .encoder_block import EncoderBlock
from .multi_head_attention import MultiHeadAttention
from .position import PositionalEncoding
from .position_wise_ffn import PositionWiseFFN
from .self_attention import SelfAttention
from .transformer import TransformerEncoder


class StreamTemporalGCN(torch.nn.Module):
    def __init__(
        self,
        input_size,
        cls_graph=None,
        graph_args=dict(),
    ):
        super(StreamTemporalGCN, self).__init__()

        C, T, V, M = input_size
        num_person = M
        in_channels = C
        num_joint = V
        num_frame=T

        if cls_graph is None:
            raise ValueError()
        else:
            self.graph = cls_graph(**graph_args)

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=3,
            kernel_size=(2, 1),
            stride=(2, 1),
            padding=(0, 0), #just on T-wise # poolint as well
            dilation=1,
            groups=1,
            bias=True,
            padding_mode="zeros",  # 'zeros', 'reflect', 'replicate', 'circular'
        )

        self.pool1 = nn.AvgPool2d(kernel_size=(2,1), stride=(2,1))

        self.transformer = TransformerEncoder(input_size=(num_frame, num_joint*in_channels),  len_seq=num_frame, dropout=0.3)

        self.conv2 = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=(3, 3), #equal padding #just on both T-wise and F-wise
            stride=(1, 1),
            padding=(1, 1),
            dilation=1,
            groups=1,
            bias=True,
            padding_mode="zeros",  # 'zeros', 'reflect', 'replicate', 'circular'
        ) 

        self.pool3 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2), padding=(0,0))  #just on both F-wise

        self.conv3 = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=(1, 3), #equal padding #just on both F-wise
            stride=(1, 1),
            padding=(0, 1),
            dilation=1,
            groups=1,
            bias=True,
            padding_mode="zeros",  # 'zeros', 'reflect', 'replicate', 'circular'
        )

        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))


    def forward(self, x):
        # stream transformer
        N_0, C, T, V, M_0 = x.size()

        # -> N-T, C, V
        stream_transformer = x.permute(0, 4, 2, 1, 3).contiguous().view(N_0*M_0, T, C, V)

        #N T, C , V => N C, T, V
        stream_transformer = stream_transformer.permute(0, 2, 1, 3)
        
        #[-1, 3, 300, 25] ->  [-1, 3, 150, 25]
        # stream_transformer = self.conv1(stream_transformer)
        stream_transformer = self.pool1(stream_transformer)
        
        # N C, T, V => N T, C , V
        stream_transformer = stream_transformer.permute(0, 2, 1, 3)
 
        N, T, C, V = stream_transformer.size()

        #[-1, 3, 150, 25]  ->  [-1, 150, 75]
        stream_transformer = stream_transformer.contiguous().view(N, T, C * V)
        #[-1, 150, 75]  -> [-1, 150, 256]
        stream_transformer = self.transformer(stream_transformer)

        #[-1, 150, 256] -> [-1, 1, 150, 256]
        stream_transformer = stream_transformer.unsqueeze(1)

        #[-1, 1, 150, 256] -> [-1, 1, 150, 256]
        stream_transformer = self.conv3(stream_transformer)

        #[-1, 1, 150, 256] -> [-1, 1, 150, 256]
        stream_transformer = self.conv3(stream_transformer)

        #[-1, 1, 150, 256] -> [-1, 1, 150, 128]
        stream_transformer = self.pool3(stream_transformer)

        #[-1, 1, 150, 128] -> [-1, 1, 150, 128]
        stream_transformer = self.conv3(stream_transformer)

        #[-1, 1, 150, 128] -> [-1, 1, 150, 128]
        stream_transformer = self.conv3(stream_transformer)

        #[-1, 1, 150, 128] -> [-1, 1, 150, 64]
        stream_transformer = self.pool3(stream_transformer)

        #[-1, 1, 150, 64] -> [-1, 1, 150, 64]
        stream_transformer = self.conv3(stream_transformer)

        #[-1, 1, 150, 64] -> [-1, 1, 150, 64]
        stream_transformer = self.conv3(stream_transformer)

        #[-1, 1, 150, 64] -> [-1, 1, 150, 32]
        stream_transformer = self.pool3(stream_transformer)
        

        #[-1, 1, 150, 256] -> [-1, 1, 150, 256]
        # stream_transformer = self.conv2(stream_transformer)

        #[-1, 1, 150, 256] -> [-1, 1, 150, 256]
        # stream_transformer = self.conv2(stream_transformer)

        #[-1, 1, 150, 256] -> [-1, 1, 75, 128]
        # stream_transformer = self.pool2(stream_transformer)

        #[-1, 1, 75, 128] -> [-1, 1, 75, 128]
        # stream_transformer = self.conv2(stream_transformer)

        #[-1, 1, 75, 128] -> [-1, 1, 75, 128]
        # stream_transformer = self.conv2(stream_transformer)

        #[-1, 1, 75, 128] -> [-1, 1, 37, 64]
        # stream_transformer = self.pool2(stream_transformer)

        #[-1, 1, 37, 64] -> [-1, 1, 37, 64]
        # stream_transformer = self.conv2(stream_transformer)

        #[-1, 1, 37, 64] -> [-1, 1, 37, 64]
        # stream_transformer = self.conv2(stream_transformer)

        #[-1, 1, 37, 64] -> [-1, 1, 18, 32]
        # stream_transformer = self.pool2(stream_transformer)

        #[-1, 1, 18, 32] -> [-1, 18, 32]
        # stream_transformer = stream_transformer.squeeze(1)
        
        # #[-1, 18, 32] -> [-1, 18]
        # stream_transformer = torch.mean(stream_transformer,dim=-1)

        # stream_transformer = stream_transformer.view(N_0, M_0, 18)
        # #why mean two people??
        # stream_transformer = stream_transformer.mean(1)

        # [-1, 1, 150, 32] -> [-1, 150, 32]
        stream_transformer = stream_transformer.squeeze(1)

        # [-1, 150, 32] -> [-1, 150]
        stream_transformer = torch.mean(stream_transformer,dim=-1)

        # [-1, 150] -> [-1/2, 2, 150]
        stream_transformer = stream_transformer.view(N_0, M_0, stream_transformer.data.size()[-1])

        # [-1/2, 2, 150] -> [-1/2, 150]
        stream_transformer = stream_transformer.mean(1)
	    
        return stream_transformer
