import torch
from os import environ

from torch import nn

from agoge import AbstractModel
from neuralDX7.models.attention import AttentionLayer
from neuralDX7.models.general import FeedForwardGELU
from neuralDX7.models.utils import position_encoding_init



class CondtionalResidualAttentionEncoder(AbstractModel):
    """
    Very similar to attention encoder but also allows custom side conditioning capacity

    """
    def __init__(self, features, c_features, attention_layer, max_len=200, n_layers=3):
        """
        features - the number of features per parameter
        c_features - the number of side conditioning features per batch item
        attention_layer - a dictionary containing instantiation parameters for the AttentionLayer module
        max_len - the maximum needed size of the positional encodings
        n_layers - number of layers for the module to use
        """
        super().__init__()


        self.layers = nn.ModuleList(
            map(lambda x: AttentionLayer(**attention_layer), range(n_layers))
        )

        positional_encoding = position_encoding_init(max_len, features)
        self.c_layers = nn.ModuleList(
            map(lambda x: FeedForwardGELU(c_features, features*2), range(n_layers))
        )

        self.p2x = nn.Linear(features, features * 2)
        self.register_buffer('positional_encoding', positional_encoding)


    def forward(self, X, A, c):
        """
        X - data tensor, torch.FloatTensor(batch_size, num_parameters, features)
        A - connection mask, torch.BoolTensor(batch_size, num_parameters, features)
        """

        # generate FiLM parameters from positional encodings for conditioning
        gamma_p, beta_p = self.p2x(self.positional_encoding).chunk(2, -1)
        gamma_p, beta_p = torch.sigmoid(gamma_p), torch.tanh(beta_p)

        X = gamma_p * X + beta_p

        for layer, c_layer in zip(self.layers, self.c_layers):

            gamma_c, beta_c = c_layer(c).chunk(2, -1)
            gamma_c, beta_c = torch.sigmoid(gamma_c), torch.tanh(beta_c)

            X = layer(gamma_c * X + beta_c, A)

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

