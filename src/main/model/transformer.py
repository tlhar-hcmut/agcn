import torch
import math

class Transformer(torch.nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        
    def __get_positional_embedding(self, N, T, F):
        # N: batch size
        # T: No of Frames
        # F: No of Feature of frame

        d_model = F  # Embedding dimension

        positional_embeddings = torch.zeros((T, d_model))

        

        # for position in range(T):
        #     for i in range(0, d_model, 2):
        #         positional_embeddings[position, i] = (
        #             math.sin(position / (10000 ** ((2*i) / d_model)))
        #         )
        #         positional_embeddings[position, i + 1] = (
        #             math.cos(position / (10000 ** ((2 * (i + 1)) / d_model)))
        #         )


        # dung vectorize tao position matrix nhanhhh
        mat_idx = torch.arange(0,T,1, dtype=torch.float).unsqueeze(0).repeat(F,1).transpose(0,1)

        mat_idx_F_x = torch.arange(0,F,2).unsqueeze(0).repeat(T,1)
        mat_idx_F_y = torch.arange(0,T,1).unsqueeze(0).repeat(F//2,1).transpose(0,1)

        mat_idx[mat_idx_F_y, mat_idx_F_x] = torch.sin(mat_idx_F_y/(10000**((mat_idx_F_x)/F)))
        mat_idx[mat_idx_F_y, mat_idx_F_x+1] = torch.cos(mat_idx_F_y/(10000**((mat_idx_F_x)/F)))

        ts_normalize_size =   mat_idx.unsqueeze(0).repeat(N,1,1)
        return ts_normalize_size

                
    def forward(self, x):
        N, C, T, V = x.size()

        mat_transpose = (
                x.permute(0, 2, 1, 3)
                .contiguous()
                .view(N, T, C * V)
        ).to("cuda")  # N-C,T,V -> N-T,C,V -> N-T,C*V

        mat_position = self.__get_positional_embedding(N, T, C * V).to("cuda")
        
        x =  mat_transpose + mat_position

        x = (
                x
                .contiguous()
                .view(N, T, C, V)
                .permute(0, 2, 1, 3)
        )  # N-T,C*V -> N-T,C,V ->     N-C,T,V 
        return x
