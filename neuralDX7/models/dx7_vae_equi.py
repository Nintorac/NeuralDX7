import torch
from torch import nn
from torch.nn import functional as F

from agoge import AbstractModel
from neuralDX7.models.equivariant_layers import ResidualEquivariantStack
from neuralDX7.models.general import FeedForwardGELU
from neuralDX7.models.stochastic_nodes import TriangularSylvesterFlow
from neuralDX7.constants import MAX_VALUE, N_PARAMS
from neuralDX7.utils import mask_parameters


class DX7VAEEquivariant(AbstractModel):
    """
    Variational Auto Encoder for a single DX7 patch. 
    
    Uses a Triangular sylvester flow to transform the encoder output to decoder input
    """

    def __init__(self, features, latent_dim, encoder, decoder, num_flows=3):
        """
        features - number of features in the model
        latent_dim - the latent dimension of the model
        encoder - dictionary containing instantiation parameters for ResidualAttentionEncoder module
        decoder - dictionary containing instantiation parameters for CondtionalResidualAttentionEncoder module
        num_flows - the number of flows for the TriangularSylvesterFlow module
        """
        
        super().__init__()

        # embedding for long param values
        self.embedder = nn.Embedding(MAX_VALUE, features)

        # equivariant encoder stack
        self.encoder = ResidualEquivariantStack(**encoder)

        # use the features from encoding stack to parameterise the posterior
        self.posterior = TriangularSylvesterFlow(features, latent_dim, num_flows)

        self.decoder = ResidualEquivariantStack(**decoder)

        self.logits = FeedForwardGELU(features, MAX_VALUE)

        self.n_features = features

    def latent_encoder(self,  X, A, z=None, mean=False):
        """
        Calculate the latent distribution

        X - data tensor, torch.FloatTensor(batch_size, 155, features)
        A - connection mask, torch.BoolTensor(batch_size, 155, features)
        z - a presampled latent, if none then the z is sampled using reparameterization technique
        mean - use the mean rather than sampling from the latent
        """
        
        return self.posterior(self.encoder(X, A).mean(-2), z)

    def forward(self, X):
        """
        Auto encodes the inputs variational latent layer

        X - the array of dx7 voices, torch.LongTensor(batch_size, 155)
        """

        # Embed long parameter to float vectors
        X_emb = self.embedder(X)

        # encode embedded parameters
        x_enc = self.encoder(X_emb).mean(-2)

        # parameterise posterior distribution
        flow = self.posterior(x_enc)
        
        # transform constant vectors given z_k
        X_dec = self.decoder(torch.ones_like(X_emb), flow.z_k)

        # get the final logits
        X_hat = self.logits(X_dec)

        return {
            'X_hat': X_hat,
            'flow': flow,
        }

if __name__ == "__main__":
    
    
    features = 100
    latent_dim = 32
    
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


    encoder_stack =  {
        'features': features, 
        'layer_args': all_layers, 
        'conditioning_dim': None, 
        'n_layers': 3
    }


    decoder_stack = {
        'features': features, 
        'layer_args': all_layers, 
        'conditioning_dim': latent_dim, 
        'n_layers': 3
    }

    model = DX7VAEEquivariant(features, latent_dim, encoder_stack, decoder_stack, num_flows=0)

    data = torch.randint(128, (10, 155))

    model(data)