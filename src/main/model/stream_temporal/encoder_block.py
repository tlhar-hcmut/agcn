from torch import nn
from .multi_head_attention import MultiHeadAttention
from .add_norm import AddNorm
from .position_wise_ffn import PositionWiseFFN

class EncoderBlock(nn.Module):
    def __init__(self, device, input_size, ffn_num_hidden, len_feature_new, len_seq, num_head, dropout, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        
        len_feature_input = input_size[-1]
        
        input_size_new = list(input_size) 
        input_size_new[-1]=len_feature_new
        input_size_new = tuple(input_size_new)

        self.attention = MultiHeadAttention(device, num_head, len_seq, len_feature_input, len_feature_new,  dropout)
        self.addnorm1 = AddNorm(input_size_new, dropout)
        self.ffn = PositionWiseFFN(len_feature_input, ffn_num_hidden, len_feature_new)

    def forward(self, X):
        Y = self.addnorm1(self.ffn(X), self.attention(X))    
        return Y