import torch
from torch import nn
from .self_attention import SelfAttention
from .attension_fusion import AttentionFusion
class MultiHeadAttention(nn.Module):
    def __init__(self, num_head, len_feature_input_mulA, len_feature_new_mulA, dropout=0, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)

        self.num_head = num_head
        
        
        self.len_feature_partials=divide_head(self.num_head, len_feature_input_mulA)
        
        self.attention = nn.ModuleList([
                            SelfAttention(len_feature_input_selfA=len_feature,
                                        len_feature_hidden_selfA=len_feature, #mat_key,query,value is square. Can be increase hidden feature size 
                                        dropout=dropout)
                            for len_feature in self.len_feature_partials])


        self.fusion1 = AttentionFusion(num_head=num_head,len_feature_input_fusion= len_feature_input_mulA,len_feature_new_fusion= len_feature_new_mulA, dropout=dropout)

    def forward(self, X, mask, output_att=False):
        X_data, att = X
        ls_output=[]
        att.append([])
        
        start   = 0
        for i in range(0,self.num_head):
            end         = start + self.len_feature_partials[i]
            attention   = self.attention[i]
            data_patial = X_data[:,:, start:end]
            output      = attention(data_patial, mask) 
            output_data, att_ = output[0], output[1]
            att[-1].append(att_)
            ls_output.append(output_data)

            start   = end
            
        return [self.fusion1(*ls_output), att]

def divide_head(num_head, len_feature):
    if len_feature%num_head == 0:
        len_feature_per_head = len_feature//num_head
        return [len_feature_per_head]*num_head
    else:
        len_feature_per_head    = len_feature//num_head
        len_feature_rest        = len_feature - num_head*len_feature_per_head
        ls_len_feature =[len_feature_per_head]*num_head
        ls_len_feature[-1]+= len_feature_rest
        return ls_len_feature
