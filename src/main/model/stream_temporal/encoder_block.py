from torch import nn
from .multi_head_attention import MultiHeadAttention
from .add_norm import AddNorm
from .res_connection import ResConnection
from .position_wise_ffn import FFN
import torch

class EncoderBlock(nn.Module):
    def __init__(self,input_size_temporal, len_feature_new, num_head, dropout, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        
        len_feature_input = input_size_temporal[-1]
        
        input_size_new = (*input_size_temporal[:-1], len_feature_new)

        self.attention = MultiHeadAttention(num_head, len_feature_input, len_feature_new,  dropout)
        self.residual1 = ResConnection(input_size_temporal, len_feature_input, len_feature_new)
        self.residual2 = ResConnection(input_size_temporal, len_feature_new, len_feature_new)

        self.addnorm1 = AddNorm(input_size_new, dropout)

        self.ffn_position = FFN(input_size_temporal = input_size_new, len_feature_input_FFN= len_feature_new,len_feature_new_FFN= len_feature_new)

    def forward(self, input):
        X, mask, output_att = input
        X_data, att = X[0], X[1]
        X_back1 = self.residual1(X_data)
        X = self.attention(X, mask, output_att)
        X_data, att = X[0], X[1]
        X_back2 = X_data + X_back1
        X_data = self.ffn_position(X_back2)
        X_data = self.addnorm1(self.residual2(X_back2), X_data)
        return ([X_data, att], mask, output_att)
