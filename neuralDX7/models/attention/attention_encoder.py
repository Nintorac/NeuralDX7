import torch
from os import environ

from torch import nn

from agoge import AbstractModel
from neuralDX7.models.attention import AttentionLayer
from neuralDX7.models.utils import position_encoding_init



class ResidualAttentionEncoder(AbstractModel):
    def __init__(self, features, attention_layer, max_len=200, n_layers=3):
        super().__init__()


        self.layers = nn.ModuleList(
            map(lambda x: AttentionLayer(**attention_layer), range(n_layers))
        )

        positional_encoding = position_encoding_init(max_len, features)

        self.p2x = nn.Linear(features, features * 2)
        self.register_buffer('positional_encoding', positional_encoding)


    def forward(self, X, A):
        gamma, beta = self.p2x(self.positional_encoding).chunk(2, -1)
        gamma = torch.sigmoid(gamma)
        beta = torch.tanh(beta)
        # beta = map(lambda f, x: f(x), fs, encodings)
        # gamma = torch.sigmoid(gamma)
        # beta = torch.tanh(beta)

        for layer in self.layers:
            X = layer(gamma * X + beta, A)

        return X
        


if __name__=='__main__':

    layer_features = 100
    n_heads = 4

    head_features = layer_features // n_heads

    attention = {
        'n_features': layer_features,
        'n_hidden': head_features,
        'n_heads': n_heads
    }
    
    attention_layer = {
        'attention': attention,
        'features': layer_features,
        'hidden_dim': 555
    }

    
    max_len = 25
    
    
    model = ResidualAttentionEncoder(layer_features, attention_layer, max_len=max_len)
    A = torch.rand(3, 25, 25)>0.5
    X = torch.distributions.Categorical(torch.ones(128)).sample((3, 25))

    model(X, A)

