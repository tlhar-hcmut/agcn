from torch import nn
class AddNorm(nn.Module):
    def __init__(self, input_size_temporal, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(input_size_temporal)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)