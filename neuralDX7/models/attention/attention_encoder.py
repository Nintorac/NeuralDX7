import torch
from os import environ

from torch import nn

from agoge import AbstractModel
from neuralDX7.models.attention import AttentionLayer
from neuralDX7.models.utils import position_encoding_init



class ResidualAttentionEncoder(AbstractModel):
    """
    Residual attention stacks based on the Attention Is All You Need paper
    """

    def __init__(self, features, attention_layer, max_len=200, n_layers=3):
        """
        features - the number of features per parameter
        c_features - the number of side conditioning features per batch item
        attention_layer - a dictionary containing instantiation parameters for the AttentionLayer module
        max_len - the maximum needed size of the positional encodings
        n_layers - number of layers for the module to use
        """
        super().__init__()

        # create the layers
        self.layers = nn.ModuleList(
            map(lambda x: AttentionLayer(**attention_layer), range(n_layers))
        )

        # pre generate the positional encodings
        positional_encoding = position_encoding_init(max_len, features)
        self.register_buffer('positional_encoding', positional_encoding)

        self.p2x = nn.Linear(features, features * 2)


    def forward(self, X, A):
        """
        X - data tensor, torch.FloatTensor(batch_size, num_parameters, features)
        A - connection mask, torch.BoolTensor(batch_size, num_parameters, features)
        """

        # generate FiLM parameters from positional encodings for conditioning
        gamma, beta = self.p2x(self.positional_encoding).chunk(2, -1)
        gamma, beta = torch.sigmoid(gamma), torch.tanh(beta)

        # Apply the data through the layers adding the positioning information in at each layer
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

