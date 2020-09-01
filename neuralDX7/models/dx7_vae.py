import torch
from torch import nn
from torch.nn import functional as F

from agoge import AbstractModel
from neuralDX7.models.attention import ResidualAttentionEncoder, CondtionalResidualAttentionEncoder
from neuralDX7.models.general import FeedForwardGELU
from neuralDX7.models.stochastic_nodes import TriangularSylvesterFlow
from neuralDX7.constants import MAX_VALUE, N_PARAMS
from neuralDX7.utils import mask_parameters


class DX7VAE(AbstractModel):
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

        self.embedder = nn.Embedding(MAX_VALUE, features)
        self.encoder = ResidualAttentionEncoder(**encoder)
        self._latent_encoder = nn.ModuleList([
            ResidualAttentionEncoder(**encoder),
            TriangularSylvesterFlow(features, latent_dim, num_flows)]
        )
        self.z_to_c = nn.Linear(latent_dim, latent_dim*155)
        self.decoder = CondtionalResidualAttentionEncoder(**decoder)
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
        
        encoder, q_x = self._latent_encoder

        return q_x(encoder(X, A).mean(-2), z)


    def forward(self, X):
        """
        Auto encodes the inputs variational latent layer

        X - the array of dx7 voices, torch.LongTensor(batch_size, 155)
        """

        batch_size = X.shape[0]

        A = torch.ones_like(X).bool()
        A = A[...,None] | A[...,None,:]

        X_emb = self.embedder(X)
        
        flow = self.latent_encoder(X_emb, A)
        
        c = self.z_to_c(flow.z_k).view(batch_size, 155, -1)

        X_dec = self.decoder(torch.ones_like(X_emb), A, c)
        X_hat = self.logits(X_dec)

        return {
            'X_hat': X_hat,
            'flow': flow,
        }


    @torch.no_grad()
    def features(self, X):
        """
        Get the latent distributions for a set of voices

        X - the array of dx7 voices, torch.LongTensor(batch_size, 155)

        """

        A = torch.ones_like(X).bool()
        A = A[...,None] | A[...,None,:]

        X = self.embedder(X)
        q = self.latent_encoder(X, A)

        return q.q_z

    @torch.no_grad()
    def generate(self, z, t=1.):
        """
        Given a sample from the latent distribution, reporojects it back to data space
        
        z - the array of dx7 voices, torch.FloatTensor(batch_size, latent_dim)
        t - the temperature of the output distribution. approaches determenistic as t->0 and approach uniforms as t->infty, requires t>0
        """
        A = z.new(z.size(0), 155, 155).bool() | 1
        X = z.new(z.size(0), 155, self.n_features)
        X = X * 0 + 1

        c = self.z_to_c(z).view(z.shape[0], 155, -1)
        X_dec = self.decoder(X, A, c)
        X_hat = mask_parameters(self.logits(X_dec))
        X_hat = torch.distributions.Categorical(logits=X_hat/t)

        return X_hat

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

    encoder = {
        'features': layer_features,
        'attention_layer': attention_layer,
        'max_len': N_PARAMS
    }
        
    
    model = DX7PatchProcess(layer_features, encoder=encoder)
    X = torch.distributions.Categorical(torch.ones(128)).sample((3, N_PARAMS))

    logits = model(X)
    print(logits.shape)
    print(logits[0])
