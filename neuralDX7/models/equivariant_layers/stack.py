import torch
from torch import nn
from os import environ

from agoge import AbstractModel
from neuralDX7.models.equivariant_layers.functions import QuadraticTransmissionLayer, LinearTransmissionLayer, AttentionEquivariantLayer, InvariantLayer
from neuralDX7.models.general import FeedForwardGELU
"""
Implements a stacked residual model based on Transformers described in Attention is all you need

allows the use of any equivariant layers in this folder
"""

Cores = {
    'quadratic': QuadraticTransmissionLayer,
    'linear': LinearTransmissionLayer,
    'attention': AttentionEquivariantLayer,
    'invariant': InvariantLayer
}


class EquivariantLayer(nn.Module):
    """
    Layer based on the original Attention is All You Need paper

    """

    def __init__(self, features, feed_forward, core, core_args, elementwise_affine=True):
        """
        feed_forward - dictionary of args for the FeedForwardGELU module
        core - the core function for the layer to compute
        core_args - instantiation dictionary for the core object
        elementwise_affine - wether or not to apply an element wise affine transform after normalisation
        """

        super().__init__()

        self.attention = Cores[core](**core_args)
        self.feedforward = FeedForwardGELU(**feed_forward)

        self.attention_norm = nn.LayerNorm(features)
        self.feedforward_norm = nn.LayerNorm(features, elementwise_affine=False) # save elementwise affine for film conditioning

    

    def forward(self, X):
        """
        X - data tensor, torch.FloatTensor(batch_size, num_parameters, features)
        A - connection mask, torch.BoolTensor(batch_size, num_parameters, features)
        """
        X = self.attention_norm(self.attention(X) + X)
        X = self.feedforward_norm(self.feedforward(X) + X)

        return X



class ResidualEquivariantStack(AbstractModel):
    """
    Residual attention stacks based on the Attention Is All You Need paper
    """

    def __init__(self, features, layer_args, conditioning_dim=None):
        """
        features - the number of features per parameter
        c_features - the number of side conditioning features per batch item
        attention_layer - a list of dictionaries containing instantiation parameters for the EquivariantLayer module
        max_len - the maximum needed size of the positional encodings
        """
        super().__init__()

        # create the layers
        self.layers = nn.ModuleList(
            map(lambda layer_arg: EquivariantLayer(**layer_arg, elementwise_affine=conditioning_dim is None), layer_args)
        )

        self.conditioning = None
        if conditioning_dim is not None:
            self.conditioning = nn.Linear(conditioning_dim, features*2)



    def forward(self, X, c=None):
        """
        X - data tensor, torch.FloatTensor(batch_size, num_parameters, features)
        c - must be supplied if conditioning_dim > 0. adds global conditioning. torch.FloatTensor(batch_size, features)
        """


        if self.conditioning is None:
            # identity scale and shift
            gamma, beta = torch.ones(X.shape), torch.zeros(X.shape)
        else:
            gamma, beta = self.conditioning(c).unsqueeze(-2).chunk(2, -1)
            

        # Apply the data through the layers adding the positioning information in at each layer
        for layer in self.layers:
            X = layer(gamma * X + beta)

        return X
        


if __name__ == "__main__":

    features = 100
    
    core_args = {
        'quadratic': lambda : {
            'input_features': features,
            'output_features': features,
            'n_heads': 4
        },
        'linear': lambda : {
            'input_features': features,
            'output_features': features,
            'n_heads': 4
        },
        'attention': lambda : {
            'input_features': features,
            'output_features': features,
            'n_heads': 4
        },
        'invariant': lambda : {
            'input_features': features,
            'output_features': features,
            'n_heads': 4,
            'max_len': 155
        }
    }

    all_layers = [{
        'features': features,
        'feed_forward': {
            'features': features
        },
        'core': core,
        'core_args': core_args[core]()

    } for core in core_args.keys()]

    equivariant_layers = [{
        'features': features,
        'feed_forward': {
            'features': features
        },
        'core': core,
        'core_args': core_args[core]()

    } for core in core_args.keys() if core!='invariant']



    
    model =  {
        'features': features, 
        'layer_args': all_layers, 
        'conditioning_dim': None, 
        'n_layers': 3
    }

    equivariant_model =  {
        'features': features, 
        'layer_args': equivariant_layers, 
        'conditioning_dim': None, 
        'n_layers': 3
    }

    cond_equivariant_model =  {
        'features': features, 
        'layer_args': equivariant_layers, 
        'conditioning_dim': 40, 
        'n_layers': 3
    }

    cond_model =  {
        'features': features, 
        'layer_args': all_layers, 
        'conditioning_dim': 40, 
        'n_layers': 3
    }


    model = ResidualEquivariantStack(**model)
    equivariant_model = ResidualEquivariantStack(**equivariant_model)
    cond_model = ResidualEquivariantStack(**cond_model)
    cond_equivariant_model = ResidualEquivariantStack(**cond_equivariant_model)

    data = torch.randn(10, 155, features)
    c = torch.randn(10, 40)
    

    p = torch.rand(155).argsort()

    assert not torch.allclose(model(data)[:,p], model(data[:,p]), atol=1e-5)
    assert torch.allclose(equivariant_model(data)[:,p], equivariant_model(data[:,p]), atol=1e-5)
    
    assert not torch.allclose(cond_model(data, c)[:,p], cond_model(data[:,p], c), atol=1e-5)
    assert torch.allclose(cond_equivariant_model(data, c)[:,p], cond_equivariant_model(data[:,p], c), atol=1e-5)
    # print((equivariant_model(data)[:,p]- equivariant_model(data[:,p])).abs().max())

    # assert torch.allclose(equivariant_model(data)[:,p], equivariant_model(data[:,p]))