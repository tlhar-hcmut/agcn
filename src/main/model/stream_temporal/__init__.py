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

        C, T, V, M = input_size
        num_person = M
        in_channels = C
        num_joint = V
        num_frame = T

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=3,
            kernel_size=(2, 1),
            stride=(2, 1),
            padding=(0, 0),  # just on T-wise # poolint as well
            dilation=1,
            groups=1,
            bias=True,
            padding_mode="zeros",  # 'zeros', 'reflect', 'replicate', 'circular'
        )

        self.pool1 = nn.AvgPool2d(kernel_size=(2, 1), stride=(2, 1))

        self.transformer = TransformerEncoder(
            input_size=(num_frame, num_joint),
            len_feature_new=len_feature_new,
            num_block=num_block,
            len_seq=num_frame,
            dropout=dropout,
            num_head=num_head,
        )

        self.conv2 = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=(3, 3),  # equal padding #just on both T-wise and F-wise
            stride=(1, 1),
            padding=(1, 1),
            dilation=1,
            groups=1,
            bias=True,
            padding_mode="zeros",  # 'zeros', 'reflect', 'replicate', 'circular'
        )

        self.pool3 = nn.MaxPool2d(
            kernel_size=(1, 2), stride=(1, 2), padding=(0, 0)
        )  # just on both F-wise

        self.conv3 = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=(1, 3),  # equal padding #just on both F-wise
            stride=(1, 1),
            padding=(0, 1),
            dilation=1,
            groups=1,
            bias=True,
            padding_mode="zeros",  # 'zeros', 'reflect', 'replicate', 'circular'
        )

        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.linear1 = nn.Linear(len_feature_new[num_block - 1], 32)

        self.ln1 = nn.LayerNorm(normalized_shape=(32))

        self.linear2 = nn.Linear(32, 1)


    def forward(self, X):
        # stream transformer
        N_0, C_0, T, V, M_0 = X.size()

        # -> NM-T, C, V
        X = X.permute(0, 4, 1, 2, 3).contiguous().view(N_0 * M_0 * C_0, T, V)

        # N T, C , V => N C, T, V
        # X = X.permute(0, 2, 1, 3)

        # [-1, 3, 300, 25] ->  [-1, 3, 150, 25]
        # X = self.conv1(X)
        # X = self.pool1(X)

        # N C, T, V => N T, C , V
        # X = X.permute(0, 2, 1, 3)

        # N, T, C, V = X.size()

        # [-1, 3, 300, 25]  ->  [-1, 300, 75]
        # X = X.contiguous().view(N, T, C * V)
        # [-1, 300, 75]  -> [-1, 300, 128]
        X = self.transformer(X)

        # [-1, 300, 128] -> [-1, 1, 300, 128]
        # X = X.unsqueeze(1)

        # #[-1, 1, 300, 128] -> [-1, 1, 300, 128]
        # X = F.relu(self.conv3(X))

        # #[-1, 1, 300, 128] -> [-1, 1, 300, 128]
        # X = F.relu(self.conv3(X))

        # #[-1, 1, 300, 128] -> [-1, 1, 300, 64]
        # X = self.pool3(X)

        # #[-1, 1, 300, 64] -> [-1, 1, 300, 64]
        # X = F.relu(self.conv3(X))

        # #[-1, 1, 300, 64] -> [-1, 1, 300, 64]
        # X = F.relu(self.conv3(X))

        # #[-1, 1, 300, 64] -> [-1, 1, 300, 32]
        # X = self.pool3(X)

        # #[-1, 1, 300, 32] -> [-1, 1, 300, 32]
        # X = F.relu(self.conv3(X))

        # #[-1, 1, 300, 32] -> [-1, 1, 300, 32]
        # X = F.relu(self.conv3(X))

        # #[-1, 1, 300, 32] -> [-1, 1, 300, 16]
        # X = self.pool3(X)

        # #[-1, 1, 300, 16] -> [-1, 1, 300, 16]
        # X = F.relu(self.conv3(X))

        # #[-1, 1, 300, 16] -> [-1, 1, 300, 16]
        # X = F.relu(self.conv3(X))

        # #[-1, 1, 300, 16] -> [-1, 1, 300, 8]
        # X = self.pool3(X)

        # #[-1, 1, 300, 8] -> [-1, 1, 300, 8]
        # X = F.relu(self.conv3(X))

        # #[-1, 1, 300, 8] -> [-1, 1, 300, 8]
        # X = F.relu(self.conv3(X))

        # #[-1, 1, 300, 8] -> [-1, 1, 300, 4]
        # X = self.pool3(X)

        # [-1, 1, 150, 256] -> [-1, 1, 150, 256]
        # X = self.conv2(X)

        # [-1, 1, 150, 256] -> [-1, 1, 150, 256]
        # X = self.conv2(X)

        # [-1, 1, 150, 256] -> [-1, 1, 75, 128]
        # X = self.pool2(X)

        # [-1, 1, 75, 128] -> [-1, 1, 75, 128]
        # X = self.conv2(X)

        # [-1, 1, 75, 128] -> [-1, 1, 75, 128]
        # X = self.conv2(X)

        # [-1, 1, 75, 128] -> [-1, 1, 37, 64]
        # X = self.pool2(X)

        # [-1, 1, 37, 64] -> [-1, 1, 37, 64]
        # X = self.conv2(X)

        # [-1, 1, 37, 64] -> [-1, 1, 37, 64]
        # X = self.conv2(X)

        # [-1, 1, 37, 64] -> [-1, 1, 18, 32]
        # X = self.pool2(X)

        # [-1, 1, 18, 32] -> [-1, 18, 32]
        # X = X.squeeze(1)

        # #[-1, 18, 32] -> [-1, 18]
        # X = torch.mean(X,dim=-1)

        # X = X.view(N_0, M_0, 18)
        # #why mean two people??
        # X = X.mean(1)

        # # [-1, 1, 300, 4] -> [-1, 300, 4]
        # X = X.squeeze(1)

        # # [-1, 300, 4] -> [-1, 300]
        # X = torch.mean(X,dim=-1)

        # [-1, 300, 128] -> [-1, 300, 32]
        X = self.linear1(X)

        X = self.ln1(X)

        X = F.relu(X)

        # [-1, 300, 32] -> [-1, 300, 1] -> [-1, 300]
        X = self.linear2(X).squeeze()

        X = X.view(N_0 * M_0, C_0, T).mean(dim=1)

        # [-1, 300] -> [-1/2, 2, 300]
        X = X.view(N_0, M_0, -1)

        # [-1, 2, 300] -> [-1/2, 300]
        # X = X[:, 0, :]
        X = X.mean(1)

        return X
