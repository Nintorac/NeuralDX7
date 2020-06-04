import torch
from torch import nn
from torch.nn import functional as F

from agoge import AbstractModel
from neuralDX7.models.attention import ResidualAttentionEncoder, CondtionalResidualAttentionEncoder
from neuralDX7.models.general import FeedForwardGELU
from neuralDX7.models.stochastic_nodes import TriangularSylvesterFlow
from neuralDX7.constants import MAX_VALUE, N_PARAMS
from neuralDX7.utils import mask_parameters

PARAMETER_MASK = mask_parameters(torch.ones(155, 128))==0
class DX7VAE(AbstractModel):

    def __init__(self, features, latent_dim, encoder, decoder, deterministic_path_drop_rate=0.5,  num_flows=3):
        
        super().__init__()

        self.PARAMETER_MASK = PARAMETER_MASK

        self.embedder = nn.Embedding(MAX_VALUE, features)
        self.encoder = ResidualAttentionEncoder(**encoder)
        self._latent_encoder = nn.ModuleList([
            ResidualAttentionEncoder(**encoder),
            TriangularSylvesterFlow(features, latent_dim, num_flows)]
        )
        self.z_to_c = nn.Linear(latent_dim, latent_dim*155)
        self.decoder = CondtionalResidualAttentionEncoder(**decoder)
        self.logits = FeedForwardGELU(features, MAX_VALUE)


        self.drop = nn.Dropout(deterministic_path_drop_rate)
        self.n_features = features

    @torch.jit.ignore
    def latent_encoder(self,  X, A, z=None, mean=False):

        encoder, q_x = self._latent_encoder

        return q_x(encoder(X, A).mean(-2), z)

    @torch.jit.ignore
    def forward(self, X):

        batch_size = X.shape[0]

        A = torch.ones_like(X)==1
        A = A[...,None] | A[...,None,:]

        X_emb = self.embedder(X)
        
        flow = self.latent_encoder(X_emb, A)
        
        c = self.z_to_c(flow.z_k).view(batch_size, 155, -1)
        # c = z.unsqueeze(-2)

        X_dec = self.decoder(torch.ones_like(X_emb), A, c)
        X_hat = self.logits(X_dec)

        return {
            'X_hat': X_hat,
            'flow': flow,
        }

    @torch.jit.ignore
    @torch.no_grad()
    def features(self, X):
        # pass
        A = torch.ones_like(X)==1
        A = A[...,None] | A[...,None,:]
        # eye = torch.eye(A.shape[-1]).bool().to(self.device) & (~X_a.unsqueeze(-2))
        # A = A | eye

        X = self.embedder(X)
        q = self.latent_encoder(X, A)

        return q.q_z

    @torch.no_grad()
    def generate(self, z, t=torch.ones(1)):
        
        A = torch.ones(z.size(0), 155, 155) == 1
        X = torch.ones(z.size(0), 155, self.n_features)
        X = X * 0 + 1

        c = self.z_to_c(z).view(z.shape[0], 155, -1)
        X_dec = self.decoder(X, A, c)
        X_hat = self.logits(X_dec)
        X_hat = X_hat.masked_fill(self.PARAMETER_MASK, -1e9)
        # X_hat = torch.distributions.Categorical(logits=X_hat/t)

        return X_hat/t

    # @torch.no_grad()
    # def generate(self, X, X_a, sample=True, t=1.):

    #     q = self.features(X, X_a)

        

    #     z = q.sample()

    #     c_gamma, c_beta = self.z_to_c(z).chunk(2, -1)


    #     X_hat = mask_parameters(self.logits(c_gamma))

    #     X_hat = torch.distributions.Categorical(logits=X_hat/t)

    #     return X_hat



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
