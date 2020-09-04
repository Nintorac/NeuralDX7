from torch import nn

class AttentionEquivariantLayer(nn.Module):
    """
    Implementation of a permutation equivariant layer in pytorch, uses a
    simple attention mechanism without positionial embeddings to achieve
    equivariance    
    """

    def __init__(self, input_features, output_features, n_heads=8, bias=True):

        super().__init__()

        assert (output_features%n_heads==0), 'output features does not \
            evenly divide number heads'

        self.QKV = nn.Linear(input_features, output_features*3)

        self.n_heads = n_heads


    def forward(self, X, *args):
        """
        X - torch.FloatTensor(b, num_nodes, in_features)
        """

        batch_size, num_nodes, *_ = X.shape

        # calculate keys and values in parallel
        QKV = self.QKV(X).view(batch_size, num_nodes, -1, 3*self.n_heads)
        # permute heads to axis 0 and split keys and values
        Q, K, V = QKV.permute(-1, 0, 1, 2).chunk(3, 0)

        # calculate outputs
        X = ((Q@K.transpose(-1, -2))/(self.n_heads**1/2)).softmax(-1) @ V

        # concatenate heads on features dim
        X = X.permute(1, 2, 3, 0).flatten(-2, -1)

        return X
