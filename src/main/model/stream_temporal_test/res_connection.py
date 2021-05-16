import torch
from torch import nn 
import torch.nn.functional as F
import src.main.util as util
class ResConnection(nn.Module):
    def __init__(self, len_feature_input_Res, len_feature_new_Res, **kwargs):
        
        super(ResConnection, self).__init__(**kwargs)
        self.len_feature_input_Res =len_feature_input_Res
        self.len_feature_new_Res = len_feature_new_Res
        
        self.dense1 = nn.Linear(len_feature_input_Res, len_feature_new_Res, bias=False)

        self.ln1    =nn.LayerNorm(normalized_shape=(len_feature_new_Res))
        util.init_bn(self.ln1, 1)

    def forward(self, X):
       
        if self.len_feature_new_Res != self.len_feature_input_Res:
            X=self.dense1(X)
            X=self.ln1(X)
        return X 
