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
        self.residual1 = ResConnection(len_feature_input, len_feature_new)
        self.residual2 = ResConnection(len_feature_new, len_feature_new)

        self.addnorm1 = AddNorm(input_size_new, dropout)

        self.ffn_position = FFN(input_size = input_size_new, len_feature_input_FFN= len_feature_new,len_feature_new_FFN= len_feature_new)

    def forward(self, X):
        X_back1 = self.residual1(X)
        X = self.attention(X)
        X_back2 = X + X_back1
        X = self.ffn_position(X_back2)
        X = self.addnorm1(self.residual2(X_back2), X)
        return X
