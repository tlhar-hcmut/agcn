import torch
from torch import nn 
import torch.nn.functional as F

activations ={"gelu":F.gelu, "relu":F.relu}
class FFN(nn.Module):
    def __init__(self, input_size_temporal, len_feature_input_FFN, len_feature_new_FFN, activation="gelu", **kwargs):
        super(FFN, self).__init__(**kwargs)

        self.dense1 = nn.Linear(len_feature_input_FFN, len_feature_new_FFN)

        len_seq     = input_size_temporal[0]
        self.ln1    =nn.LayerNorm(normalized_shape=(len_seq, len_feature_new_FFN))
        
        if activation not in activations:
            act = activations["gelu"]
        else:
            act = activations[activation]
        self.activation = act

    def forward(self, X):
        
        X = self.dense1(X)
        
        X = self.ln1(X)

        X = self.activation(X)
        
        return X 
