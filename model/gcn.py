import torch
import numpy as np
from model import tools


class UnitGCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, mat_adj, coff_embedding=4, num_subset=3):
        super(UnitGCN, self).__init__()
        # Init constant
        self.num_subset = num_subset
        self.inter_channels = out_channels // coff_embedding
        # Init cnn layer
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU()
        self.soft = torch.nn.Softmax(-2)
        # Init gcn layler
        self.mat_adj = torch.autograd.Variable(
            data=torch.from_numpy(mat_adj.astype(np.float32)),
            requires_grad=False,
        )
        self.weight = torch.nn.Parameter(
            data=torch.from_numpy(mat_adj.astype(np.float32)),
            requires_grad=True,
        )
        torch.nn.init.constant_(self.weight, 1e-6)
        # Init embedding layer
        self.conv_a = torch.nn.ModuleList()
        self.conv_b = torch.nn.ModuleList()
        self.conv_d = torch.nn.ModuleList()
        for i in range(self.num_subset):
            self.conv_a.append(torch.nn.Conv2d(in_channels, self.inter_channels, 1))
            self.conv_b.append(torch.nn.Conv2d(in_channels, self.inter_channels, 1))
            self.conv_d.append(torch.nn.Conv2d(in_channels, out_channels, 1))

        if in_channels != out_channels:
            self.conv_res = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, 1),
                torch.nn.BatchNorm2d(out_channels),
            )
        else:
            self.conv_res = lambda x: x

        # Init weight
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                tools.init_conv(m)
            elif isinstance(m, torch.nn.BatchNorm2d):
                tools.init_bn(m, 1)

        tools.init_bn(self.bn, 1e-6)

        for i in range(self.num_subset):
            tools.init_conv_branch(self.conv_d[i], self.num_subset)

    def forward(self, x):
        N, C, T, V = x.size()
        mat_adj = self.mat_adj.cuda(x.get_device())
        mat_adj = mat_adj + self.weight

        y = None
        for i in range(self.num_subset):
            # Embed spatial channel to embedding channels
            mat_embed_1 = self.conv_a[i](x)\
                .permute(0, 3, 1, 2).contiguous()\
                .view(N, V, self.inter_channels * T) # N-C,T,V -> N-V,T,C -> N-V,TC

            mat_embed_2 = self.conv_b[i](x)\
                .view(N, self.inter_channels * T, V) # N-C,T,V -> N-CT,V
            # Build adaptive adjacency matrix
            mat_enhance = self.soft(torch.matmul(mat_embed_1, mat_embed_2) / A1.size(-1))  # N-V,V
            mat_inpt = x.view(N, C * T, V) # N-CT,V
            mat_adapt = mat_adj[i] + mat_enhance # N-V,V
            z = self.conv_d[i](torch.matmul(mat_inpt, mat_adapt).view(N, C, T, V)) # N-C,T,V
            y = z + y if y is not None else z

        return self.relu(self.bn(y) + self.conv_res(x))


if __name__ == "__main__":
    model = UnitGCN(3, 64, np.ones((25, 25)))
    print(model)