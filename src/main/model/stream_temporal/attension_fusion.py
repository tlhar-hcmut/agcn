from torch import nn
import torch
import torch.nn.functional as F

class AttentionFusion(nn.Module):
    def __init__(self, input_size, dropout, **kwargs):
        super(AttentionFusion, self).__init__(**kwargs)
        self.input_size = input_size
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(input_size)

        self.W_o = nn.Linear(input_size[-1]*input_size[-2], input_size[-1])


    def forward(self, *X):
        X_out = torch.empty(X[0].size()).to(X[0].get_device())
        
        for X_ in X:
            X_out+=self.dropout(X_)

        return X_out



        # output_concat = torch.cat(X, dim=1)

        # # N, num_head*len_seq, len_feature => N, len_feature, num_head*len_seq
        # output_concat = output_concat.permute(0, 2, 1)
        # output_concat = self.W_o(output_concat)

        # #  N, len_feature, len_seq => N, len_seq, len_feature
        # output_concat = output_concat.permute(0, 2, 1)
        # return self.ln(output_concat)