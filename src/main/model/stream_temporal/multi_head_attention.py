import torch
from torch import nn
from .self_attention import SelfAttention
from .attension_fusion import AttentionFusion
class MultiHeadAttention(nn.Module):
    def __init__(self, num_head, len_feature_input_mulA, len_feature_new_mulA, dropout=0, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)

        self.num_head = num_head

        self.attention = nn.ModuleList([
                            SelfAttention(len_feature_input_selfA=len_feature_input_mulA,
                                        len_feature_hidden_selfA=len_feature_input_mulA, #mat_key,query,value is square. Can be increase hidden feature size 
                                        dropout=dropout)
                            for _ in range(num_head)])


        self.fusion1 = AttentionFusion(num_head=num_head,len_feature_input_fusion= len_feature_input_mulA,len_feature_new_fusion= len_feature_new_mulA, dropout=dropout)

    def forward(self, X):
        ls_output = [self.attention[i](X) for i in range(self.num_head)]
        return self.fusion1(*ls_output)
