import torch
from torch import nn 
import torch.nn.functional as F

class PositionWiseFFN(nn.Module):
    def __init__(self,len_feature_input, ffn_num_hiddens, len_feature_new, **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(len_feature_input, ffn_num_hiddens)
        self.dense2 = nn.Linear(ffn_num_hiddens, len_feature_new)

    def forward(self, X):
        return self.dense2(F.relu(self.dense1(X)))