import numpy as np
import torch
from torch import nn
class PositionalEncoding(nn.Module):

    def __init__(self, len_feature_input, dropout, max_len=1000, device="cuda"):
        super(PositionalEncoding, self).__init__()
        self.device =device
        self.dropout = nn.Dropout(dropout)
        # Create a long enough P
        self.P = torch.zeros((1, max_len, len_feature_input))
        X = torch.arange(0, max_len).reshape(-1, 1) / torch.pow(10000, torch.arange(0, len_feature_input, 2) / len_feature_input)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X) if len_feature_input%2==0 else torch.cos(X[:,:-1])

    def forward(self, X, mask):
        X_view_flatten = X.view(X.shape[0],-1)
        Max_X,_ = torch.max(X_view_flatten, dim=-1,keepdim=True)
        Min_X,_ = torch.min(X_view_flatten, dim=-1, keepdim=True)
        #scale into [0,1] for position adding
        X_view_flatten= (X_view_flatten-Min_X)/(Max_X-Min_X)
        X =  X_view_flatten.reshape(X.shape)
        X = X + self.P[:, :X.shape[1], :].to(self.device)
        
        #after addition position value, we still reset value padding to 0
        mask_float = mask.float()
        mask_float = mask_float.unsqueeze(-1) #up dimention
        X = X*mask_float

        return self.dropout(X) 