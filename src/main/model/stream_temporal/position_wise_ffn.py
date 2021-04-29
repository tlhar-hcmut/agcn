import torch
from torch import nn 
import torch.nn.functional as F

class PositionWiseFFN(nn.Module):
    def __init__(self,len_feature_input, ffn_num_hiddens, len_feature_new, **kwargs):
        
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.ffn_num_hiddens = ffn_num_hiddens
        self.dense1 = nn.Linear(len_feature_input, len_feature_new)
        if self.ffn_num_hiddens == 0: return
        self.dense2 = nn.Linear(len_feature_input, ffn_num_hiddens)
        self.dense3 = nn.Linear(ffn_num_hiddens, len_feature_new)

    def forward(self, X):
        if (self.ffn_num_hiddens>0):
            X = self.dense3(F.relu(self.dense2(X)))
        else:
            X = F.relu(self.dense1(X))

        return X 
