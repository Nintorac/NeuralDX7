import torch
from torch import nn
from torch.distributions import Normal



class NormalNode(nn.Module):

    def __init__(self, in_features, latent_dim, hidden_dim=None):
        super().__init__()

        if hidden_dim is None:

            hidden_dim = in_features * 2

        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim * 2)
        )

    @torch.jit.ignore
    def forward(self, x):


        mu, sigma = self.net(x).chunk(2, -1)

        p = Normal(mu, (sigma*0.5).clamp(-5, 4).exp())

        return p



