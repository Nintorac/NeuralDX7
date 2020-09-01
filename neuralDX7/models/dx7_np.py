import torch
from torch import nn
from torch.nn import functional as F

from agoge import AbstractModel
from neuralDX7.models.attention import ResidualAttentionEncoder, CondtionalResidualAttentionEncoder
from neuralDX7.models.general import FeedForwardGELU
from neuralDX7.models.stochastic_nodes import NormalNode
from neuralDX7.constants import MAX_VALUE, N_PARAMS
from neuralDX7.utils import mask_parameters


class DX7NeuralProcess(AbstractModel):
    """
    EXPERIMENTAL AND UNTESTED

    """

    def __init__(self, features, latent_dim, encoder, decoder, deterministic_path_drop_rate=0.5):
        
        super().__init__()

        self.embedder = nn.Embedding(MAX_VALUE, features)
        self.encoder = ResidualAttentionEncoder(**encoder)
        self._latent_encoder = nn.ModuleList([
            ResidualAttentionEncoder(**encoder),
            NormalNode(features, latent_dim)]
        )
        self.z_to_c = nn.Linear(latent_dim, latent_dim*155)
        self.decoder = CondtionalResidualAttentionEncoder(**decoder)
        self.logits = FeedForwardGELU(features, MAX_VALUE)
        self.drop = nn.Dropout(deterministic_path_drop_rate)

    def latent_encoder(self,  X, A, mean=False):

        encoder, q_x = self._latent_encoder

        return q_x(encoder(X, A).mean(-2))


    def forward(self, X):

        # generate random masks
        batch_p = torch.rand(X.shape[0]) # decide p value for each item in batch
        item_logits = torch.rand(X.shape) # random value for each param
        X_a = batch_p.unsqueeze(-1) <= item_logits # active params in X
        X_a = X_a.to(self.device)

        A = (~X_a.unsqueeze(-1)) & (X_a.unsqueeze(-2))
        eye = torch.eye(A.shape[-1]).bool().to(self.device) & (~X_a.unsqueeze(-2))
        A = A | eye
        # A = A | True

        X_target = self.embedder(X)
        
        X_context = X_target * X_a.unsqueeze(-1).float()

        q_context = self.latent_encoder(X_context, A)
        q_target = self.latent_encoder(X_target, A | (~X_a.unsqueeze(-1)))

        # r = self.drop(self.encoder(X_context, A))
        # X_encoded = F.drop out
        z = q_target.rsample()
        # z_context = q_context.rsample()
        # mask = (torch.rand_like(z_target[...,[0]]) > 0.5).float()
        # z = (z_target * mask) + (z_context * (1-mask))

        c = self.z_to_c(z).view(z.shape[0], 155, -1)
        # c = z.unsqueeze(-2)

        X_dec = self.decoder(X_context, A, c)
        X_hat = self.logits(X_dec)

        return X_hat, X_a, q_context, q_target, z

    @torch.no_grad()
    def features(self, X, X_a):

        A = (~X_a.unsqueeze(-1)) & (X_a.unsqueeze(-2))
        eye = torch.eye(A.shape[-1]).bool().to(self.device) & (~X_a.unsqueeze(-2))
        A = A | eye

        X = self.embedder(X) * X_a.unsqueeze(-1).float()
        q = self.latent_encoder(X, A)

        return q

    @torch.no_grad()
    def generate_z(self, X, X_a, z, t=1.):


        A = (~X_a.unsqueeze(-1)) & (X_a.unsqueeze(-2))
        eye = torch.eye(A.shape[-1]).bool().to(self.device) & (~X_a.unsqueeze(-2))
        A = A | eye

        X = self.embedder(X)
        X = X * X_a.unsqueeze(-1).float()       

        c = self.z_to_c(z).view(z.shape[0], 155, -1)
        X_dec = self.decoder(X, A, c)
        X_hat = mask_parameters(self.logits(X_dec))
        X_hat = torch.distributions.Categorical(logits=X_hat/t)

        return X_hat

    @torch.no_grad()
    def generate(self, X, X_a, sample=True, t=1.):

        q = self.features(X, X_a)

        

        z = q.sample()

        c_gamma, c_beta = self.z_to_c(z).chunk(2, -1)


        X_hat = mask_parameters(self.logits(c_gamma))

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
