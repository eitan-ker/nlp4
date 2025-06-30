from torch import nn

class MLP(nn.Module):
    def __init__(self, inout_dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.i2h = nn.Linear(inout_dim, hidden_dim)
        self.h2o = nn.Linear(hidden_dim, inout_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        x = self.i2h(inputs)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.h2o(x)
        return x
