import torch
from torch import nn

from neuralDX7.models.attention import Attention



class AttentionLayer(nn.Module):


    def __init__(self, features, hidden_dim, attention):
        """
        features - the number of features the layer has at input and output
        hidden_dim - the hidden dimension of the feedforward network

        """

        super().__init__()

        self.attention = Attention(**attention)
        self.feedforward = nn.Sequential(
            nn.Linear(features, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, features),
        )

        self.attention_norm = nn.LayerNorm(features)
        self.feedforward_norm = nn.LayerNorm(features, elementwise_affine=False)

    

    def forward(self, X, A):

        X = self.attention_norm(self.attention(X, A) + X)
        X = self.feedforward_norm(self.feedforward(X) + X)

        return X

if __name__=="__main__":

    attention = {
        'n_features': 100,
        'n_hidden': 25,
        'n_heads': 4
    }
    
    model = AttentionLayer(100, 250, attention)

    X = torch.randn(3, 25, 100)
    A = torch.rand(3, 25, 25)>0.5
    Y = model(X, A)

    print(Y.shape)