import torch
from torch import nn

class LinearTransmissionLayer(nn.Module):
    """
    PyTorch implementation of the Linear Equivariant function as specified
    in "On Universal Equivariant Set Networks"
    https://arxiv.org/abs/1910.02421
    
    """

    def __init__(self, input_features, output_features, n_heads=8, bias=True):

        super().__init__()

        assert (output_features%n_heads==0), 'n_heads needs to divide output features exactly'
        self.i2h = nn.Linear(input_features, output_features)

        head_dim = output_features//n_heads

        self.A = nn.Parameter(torch.randn(n_heads, 1, head_dim, head_dim))
        self.B = nn.Parameter(torch.randn(n_heads, 1, head_dim, head_dim))
        self.c = nn.Parameter(torch.zeros(n_heads, 1, 1, head_dim), requires_grad=bias)

        self.n_heads = n_heads


    def forward(self, X, *args):
        """
        X - torch.FloatTensor(b, num_nodes, in_features)
        """

        batch_size, num_nodes, _ = X.shape
        
        # compute head features
        X = self.i2h(X).view(batch_size, num_nodes, -1, self.n_heads)
        X = X.permute(-1, 0, 1, 2)

        # compute layer transmission
        X = (X @ self.A) + (X @ self.B).mean(-2, keepdim=True) + self.c

        # concat heads on features
        X = X.permute(1, 2, 3, 0).flatten(-2, -1)

        return X

