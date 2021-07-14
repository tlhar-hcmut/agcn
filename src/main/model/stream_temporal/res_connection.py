import torch
from torch import nn 
import torch.nn.functional as F

class ResConnection(nn.Module):
    def __init__(self,input_size_temporal, len_feature_input_Res, len_feature_new_Res, **kwargs):
        
        super(ResConnection, self).__init__(**kwargs)
        self.len_feature_input_Res =len_feature_input_Res
        self.len_feature_new_Res = len_feature_new_Res
        
        self.dense1 = nn.Linear(len_feature_input_Res, len_feature_new_Res, bias=False)

        len_seq     = input_size_temporal[0]
        self.ln1    =nn.LayerNorm(normalized_shape=(len_seq, len_feature_new_Res))

    def forward(self, X):
       
        if self.len_feature_new_Res != self.len_feature_input_Res:
            X=self.dense1(X)
            X=self.ln1(X)
        return X 
