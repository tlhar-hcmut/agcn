from torch import nn
import torch
import torch.nn.functional as F
import src.main.util as util



class AttentionFusion(nn.Module):
    def __init__(self, len_feature_input_fusion, len_feature_new_fusion, num_head=3, dropout=0, **kwargs):
        super(AttentionFusion, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

        self.W_o = nn.Linear(len_feature_input_fusion, len_feature_new_fusion)
        self.ln1 = nn.LayerNorm(len_feature_new_fusion)

        util.init_bn(self.ln1, 1)
        
    def forward(self, *X):
        output_concat = torch.cat(X, dim=-1)
        output_concat = self.W_o(output_concat)
        output_concat = self.ln1(output_concat)
        output_concat = self.dropout(output_concat)
        return output_concat
