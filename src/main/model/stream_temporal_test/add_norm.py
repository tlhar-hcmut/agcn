from torch import nn
import src.main.util as util
class AddNorm(nn.Module):
    def __init__(self, input_size_temporal, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln1 = nn.LayerNorm(input_size_temporal[-1])

        util.init_bn(self.ln1, 1)


    def forward(self, X, Y):
        return self.ln1(self.dropout(Y) + X)