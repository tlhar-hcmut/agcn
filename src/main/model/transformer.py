import torch
import math

class TransformerUnit(torch.nn.Module):
    def __init__(self, T, F=75):
        '''
            V: length of sentense (num of frame)
            F: dimention of feature input
        '''
        self.T=T
        self.F = F
        super(TransformerUnit, self).__init__()

        #embed dimension F-> F_out

        self.weight_q = torch.ones((F,F), requires_grad=True).to('cuda')
        self.weight_k = torch.ones((F,F), requires_grad=True).to('cuda')
        self.weight_v = torch.ones((F, F), requires_grad=True).to('cuda')
        
        self.sm1 = torch.nn.Softmax(0)


    def __get_positional_embedding(self):
        # N: batch size
        # T: No of Frames
        # F: No of Feature of frame

        # d_model = self.F  # Embedding dimension
        # positional_embeddings = torch.zeros((self.T, d_model))
        # for position in range(T):
        #     for i in range(0, d_model, 2):
        #         positional_embeddings[position, i] = (
        #             math.sin(position / (10000 ** ((2*i) / d_model)))
        #         )
        #         positional_embeddings[position, i + 1] = (
        #             math.cos(position / (10000 ** ((2 * (i + 1)) / d_model)))
        #         )

        # dung vectorize tao position matrix nhanhhh
        mat_idx = torch.arange(0,self.T,1, dtype=torch.float).unsqueeze(0).repeat(self.F,1).transpose(0,1)

        mat_idx_F_x = torch.arange(0,self.F,2).unsqueeze(0).repeat(self.T,1)
        mat_idx_F_y = torch.arange(0,self.T,1).unsqueeze(0).repeat((self.F+1)//2,1).transpose(0,1)

        mat_idx[mat_idx_F_y, mat_idx_F_x] = torch.sin(mat_idx_F_y/(10000**((mat_idx_F_x)/self.F)))
        mat_idx[mat_idx_F_y[:,:-1], mat_idx_F_x[:,:-1]+1] = torch.cos(mat_idx_F_y[:,:-1]/(10000**((mat_idx_F_x[:,:-1])/self.F)))
        if (self.F%2==0):
            mat_idx[mat_idx_F_y[:, -1:], mat_idx_F_x[:, -1:]+1] = torch.cos(mat_idx[mat_idx_F_y[:, -1:]]/(10000**((mat_idx_F_x[:, -1:])/self.F)))

        # ts_normalize_size =   mat_idx.unsqueeze(0).repeat(N,1,1)

        

        return mat_idx

    def forward(self, x):
        N, T, F = x.size()
        
        #position embedding
        mat_position = self.__get_positional_embedding().to("cuda")
        mat_position =   mat_position.unsqueeze(0).repeat(N,1,1)

        x =  x+ mat_position

        #self-attention
       
        mat_q = torch.matmul(x, self.weight_q)
        mat_k = torch.matmul(x, self.weight_k)
        mat_v = torch.matmul(x, self.weight_v)
        mat_kT = mat_k.permute(0,2,1)
        mat_attention_score = torch.matmul(mat_q, mat_kT)

        mat_attention_score = self.sm1(mat_attention_score)

        return torch.matmul(mat_attention_score, mat_v)
