import torch
from torch import nn

class InvariantLayer(nn.Module):
    """
    Implementation of a permutation invariant layer in pytorch, uses a
    simple attention mechanism with fixed queries to obtain equivariance
    https://arxiv.org/abs/1910.02421
    
    """

    def __init__(self, input_features, output_features, n_heads=8, bias=True, max_len=155):

        super().__init__()

        assert (output_features%n_heads==0), 'output features does not \
            evenly divide number heads'

        self.Q = nn.Parameter(torch.randn(1, max_len, output_features//n_heads))
        self.KV = nn.Linear(input_features, output_features*2)

        self.n_heads = n_heads


    def forward(self, X, *args):
        """
        X - torch.FloatTensor(b, num_nodes, in_features)
        """

        batch_size, num_nodes, *_ = X.shape

        # calculate keys and values in parallel
        KV = self.KV(X).view(batch_size, num_nodes, -1, 2*self.n_heads)
        # permute heads to axis 0 and split keys and values
        K, V = KV.permute(-1, 0, 1, 2).chunk(2, 0)

        # calculate outputs
        X = ((self.Q @ K.transpose(-1, -2))/(self.n_heads**1/2)).softmax(-1) @ V

        # concatenate heads on features dim
        X = X.permute(1, 2, 3, 0).flatten(-2, -1)

        return X
