import torch
from torch import nn
import math
from .position import PositionalEncoding
from .encoder_block import EncoderBlock


class TransformerEncoder(torch.nn.Module):
    def __init__(self, device, input_size,  ffn_num_hiddens=[128,256,512], len_feature_new=[64, 128, 256], len_seq =300, num_head=5,  dropout=0, num_block=3, **kwargs):

        super(TransformerEncoder, self).__init__(**kwargs)
        self.len_feature_input= input_size[-1]
        self.len_feature_new = len_feature_new
        self.pos_encoding = PositionalEncoding(self.len_feature_input, dropout)

        layer=[]
        for i in range(num_block):
            layer.append(EncoderBlock(device, input_size, ffn_num_hiddens[i], len_feature_new[i], len_seq, num_head, dropout))
            input_size =list(input_size)
            input_size[-1]=len_feature_new[i]
            input_size = tuple(input_size)
        self.blks = nn.Sequential(*layer)

    def forward(self, X):
        X = self.pos_encoding(X * math.sqrt(self.len_feature_input))
        X = self.blks(X)
        return X