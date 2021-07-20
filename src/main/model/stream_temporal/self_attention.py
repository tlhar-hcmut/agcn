import torch
from torch import nn 
import torch.nn.functional as F
import math

class  SelfAttention(nn.Module):
    def __init__(self,len_feature_input_selfA, len_feature_hidden_selfA, dropout = 0, **kwargs):

        super(SelfAttention, self).__init__(**kwargs)
        self.len_feature_hidden_selfA = len_feature_hidden_selfA
        self.W_k =nn.Linear(len_feature_input_selfA, len_feature_hidden_selfA, bias=False)
        self.W_q =nn.Linear(len_feature_input_selfA, len_feature_hidden_selfA, bias=False)
        self.W_v = nn.Linear(len_feature_input_selfA, len_feature_hidden_selfA, bias=False)

        self.dropout = nn.Dropout(dropout)
    def forward(self, X, mask):
        K = self.W_k(X)
        Q = self.W_q(X)
        V = self.W_v(X)

        K_T = K.permute(0,2,1)
        scores = torch.matmul(Q,K_T)/math.sqrt(self.len_feature_hidden_selfA)
        mask_1 = mask.float().unsqueeze(-1)
        mask_2 = mask.float().unsqueeze(1)
        mask =mask_1 * mask_2
        scores = F.softmax(scores, -1)
        scores = scores.masked_fill(mask==0.0, 10e-10)

        scores = self.dropout(scores)
        
        #len_seq x len_feature_hidden_selfA
        attention = torch.matmul(scores, V)
        return attention, scores


        
