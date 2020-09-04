import torch
from torch import nn

class QuadraticTransmissionLayer(nn.Module):
    """
    PyTorch implementation of the Quadratic Equivariant function as specified
    in "On Universal Equivariant Set Networks"
    https://arxiv.org/abs/1910.02421
    
    """
    def __init__(self, input_features, output_features, bias=True, n_heads=8):

        super().__init__()

        self.i2h = nn.Linear(input_features, output_features)

        assert (output_features%n_heads==0), 'output features does not \
            evenly divide number heads'

        head_dim = output_features//n_heads

        self.A = nn.Parameter(torch.randn(n_heads, 1, head_dim, head_dim))
        self.B = nn.Parameter(torch.randn(n_heads, 1, head_dim, head_dim))
        self.C = nn.Parameter(torch.randn(n_heads, 1, head_dim, head_dim))
        self.D = nn.Parameter(torch.randn(n_heads, 1, head_dim, head_dim))
        self.E = nn.Parameter(torch.randn(n_heads, 1, head_dim, head_dim))

        self.bias = nn.Parameter(torch.zeros(n_heads, 1, 1, head_dim), requires_grad=bias)

        self.n_heads = n_heads

    def forward(self, X, *args):
        """
        X - torch.FloatTensor(b, num_nodes, in_features)
        """

        batch_size, num_nodes, _ = X.shape

        # compute head features
        X = self.i2h(X).view(batch_size, num_nodes, -1, self.n_heads)
        X = X.permute(-1, 0, 1, 2)

        X =     X @ self.A + \
                (X-X.mean(-2, keepdim=True)) @ self.B + \
                (X.mean(-2, keepdim=True)**2) @ self.C + \
                (X**2).mean(-2, keepdim=True) @ self.D + \
                (X.mean(2, keepdim=True) * X) @ self.E + \
                self.bias

        # concat heads on features
        X = X.permute(1, 2, 3, 0).flatten(-2, -1)

        return X