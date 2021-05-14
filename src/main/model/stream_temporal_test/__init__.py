import src.main.config.cfg_train as cfg_train
import torch
import torch.nn.functional as F
from torch import nn

from .transformer import TransformerEncoder


class StreamTemporalGCN(torch.nn.Module):
    def __init__(
        self, input_size, num_head, num_block, dropout, len_feature_new, **kargs
    ):
        super(StreamTemporalGCN, self).__init__()

        input_size = (25, 300, 16, 2)
        C, T, V, M = input_size
        num_person = M
        in_channels = C
        num_joint = V
        num_frame = T

        #conv 1x1
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=8,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),  
            dilation=1,
            groups=1,
            bias=True,
            padding_mode="zeros",  # 'zeros', 'reflect', 'replicate', 'circular'
        )

        self.ln0 = nn.LayerNorm(normalized_shape=(num_frame,num_joint))

        #conv 1x1
        self.conv2 = nn.Conv2d(
            in_channels=8,
            out_channels=8,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),  
            dilation=1,
            groups=1,
            bias=True,
            padding_mode="zeros",  # 'zeros', 'reflect', 'replicate', 'circular'
        )

        # self.pool1 = nn.AvgPool2d(kernel_size=(2, 1), stride=(2, 1))

        self.transformer = TransformerEncoder(
            input_size=(num_frame, num_joint),
            len_feature_new=len_feature_new,
            num_block=num_block,
            len_seq=num_frame,
            dropout=dropout,
            num_head=num_head,
        )

        # self.conv2 = nn.Conv2d(
        #     in_channels=1,
        #     out_channels=1,
        #     kernel_size=(3, 3),  # equal padding #just on both T-wise and F-wise
        #     stride=(1, 1),
        #     padding=(1, 1),
        #     dilation=1,
        #     groups=1,
        #     bias=True,
        #     padding_mode="zeros",  # 'zeros', 'reflect', 'replicate', 'circular'
        # )

        # self.pool3 = nn.MaxPool2d(
        #     kernel_size=(1, 2), stride=(1, 2), padding=(0, 0)
        # )  # just on both F-wise

        # self.conv3 = nn.Conv2d(
        #     in_channels=1,
        #     out_channels=1,
        #     kernel_size=(1, 3),  # equal padding #just on both F-wise
        #     stride=(1, 1),
        #     padding=(0, 1),
        #     dilation=1,
        #     groups=1,
        #     bias=True,
        #     padding_mode="zeros",  # 'zeros', 'reflect', 'replicate', 'circular'
        # )

        # self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.linear1 = nn.Linear(len_feature_new[num_block - 1], 32)

        self.ln1 = nn.LayerNorm(normalized_shape=(300,32))

        self.linear2 = nn.Linear(32, 1)

        self.linear3 = nn.Linear(8, 1)


    def forward(self, X):

        N_0, C_0, T, V, M_0 = X.size()

        #process parralell two bodies
        X = X.permute(0, 4, 1, 2, 3).contiguous().view(N_0 * M_0, C_0 , T, V)
        
        # embed 3 channels into 8 channels => or be feeded by spatial part
        # X = self.conv1(X)

        # X = self.ln0(X)

        # X = F.relu(X)
        
        # X = self.conv2(X)


        N_1, C_1, T, V= X.size()

        # -> NMC-T, V
        X = X.view(N_1*C_1, T, V)

        # [-1, 3, 300, 25]  ->  [-1, 300, 75]
        # X = X.contiguous().view(N, T, C * V)
        # [-1, 300, 75]  -> [-1, 300, 128]
        X = self.transformer(X)

        # [-1, 300, 128] -> [-1, 300, 32]
        X = self.linear1(X)

        X = self.ln1(X)

        X = F.relu(X)

        # [-1, 300, 32] -> [-1, 300, 1] -> [-1, 300]
        # X = self.linear2(X).squeeze()
        X = X.mean(-1).squeeze()
        # [-1, C , T] -> [-1, T, C]
        X = X.view(N_1, C_1, T).permute(0,2,1)

        # X = self.linear3(X).squeeze()
        X = X.mean(-1).squeeze()

        # [-1, 300] -> [-1/2, 2, 300]
        X = X.view(N_0, M_0, -1)

        # [-1, 2, 300] -> [-1/2, 300]
        # X = X[:, 0, :]
        X = X.mean(1)

        return X