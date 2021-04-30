from torch import nn
import torch
import torch.nn.functional as F

class AttentionFusion(nn.Module):
    def __init__(self, input_size, num_head=3, dropout=0, **kwargs):
        super(AttentionFusion, self).__init__(**kwargs)
        self.input_size = input_size
        # self.dropout = nn.Dropout(dropout)
        # self.ln = nn.LayerNorm(input_size)

        len_feature = input_size[-1]
        self.W_o = nn.Linear(num_head*len_feature, len_feature)


    def forward(self, *X):
        # X_out = torch.empty(X[0].size()).to(X[0].get_device())
        
        # for X_ in X:
        #     X_out+=self.dropout(X_)

        # return X_out

        # N, len_seq, num_head*len_feature
        output_concat = torch.cat(X, dim=-1)
        output_concat = self.W_o(output_concat)

        return output_concat
