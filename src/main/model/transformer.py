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

        
        # dung vectorize tao position matrix nhanhhh

        for position in range(T):
            for i in range(0, d_model, 2):
                positional_embeddings[position, i] = (
                    math.sin(position / (10000 ** ((2*i) / d_model)))
                )
                positional_embeddings[position, i + 1] = (
                    math.cos(position / (10000 ** ((2 * (i + 1)) / d_model)))
                )
        return torch.cat(N*[positional_embeddings])

                
    def forward(self, x):
        N, C, T, V = x.size()

        mat_transpose = (
                x.permute(0, 2, 1, 3)
                .contiguous()
                .view(N, T, C * V)
        )  # N-C,T,V -> N-T,C,V -> N-T,C*V

        mat_position = self.__get_positional_embedding(N, T, C * V)
        
        x =  mat_transpose + mat_position

        x = (
                x
                .view(N, T, C, V)
                .contiguous()
                .permute(0, 2, 1, 3)
        )  # N-T,C*V -> N-T,C,V ->     N-C,T,V 
        return x
