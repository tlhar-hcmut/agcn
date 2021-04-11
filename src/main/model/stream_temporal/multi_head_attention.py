import torch
from torch import nn 
from .self_attention import SelfAttention

class MultiHeadAttention(nn.Module):
    def __init__(self,device,  num_head, len_seq, len_feature_input, len_feature_new, dropout=0, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)

        self.num_head = num_head
        self.attention = [SelfAttention(device, len_feature_input = len_feature_input, len_feature_new=len_feature_new, dropout=dropout) for _ in range(num_head)]
        self.W_o = nn.Linear(64, len_seq)
    def forward(self, X):
        ls_output = [self.attention[i](X) for i in range(self.num_head)]
        output_concat = torch.cat(ls_output, dim=1)

        A= self.W_o(output_concat)
        return A


