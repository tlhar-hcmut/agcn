from torch import nn
from .multi_head_attention import MultiHeadAttention
from .add_norm import AddNorm
from .res_connection import ResConnection
from .position_wise_ffn import FFN

class EncoderBlock(nn.Module):
    def __init__(self,input_size, len_feature_new, num_head, dropout, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        
        len_feature_input = input_size[-1]
        
        input_size_new = (*input_size[:-1], len_feature_new)

        self.attention = MultiHeadAttention(num_head, len_feature_input, len_feature_new,  dropout)
        self.residual = ResConnection(len_feature_input, len_feature_new)
        self.addnorm1 = AddNorm(input_size_new, dropout)

        self.ffn_position = FFN(len_feature_new, len_feature_new)
    def forward(self, X):
        X_back = X
        X = self.attention(X)
        X = self.ffn_position(X)
        X = self.addnorm1(self.residual(X_back), X)
        return X
