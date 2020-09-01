from torch import nn
from torch.distributions import Normal



class NormalNode(nn.Module):
    """
    Simple module to create a normally distributed node in a ala VAE's. 

    this node computes the function
    ```
        p(x) = N(mu(x), sigma(x)I)
    ```
    """

    def __init__(self, in_features, latent_dim, hidden_dim=None):
        """
        in_features - number of input features
        latent_dim - number of normals in the output
        hidden_dim - the inner dimension of the nonlinear feedforward network, 2x the input dimension if None
        """
        super().__init__()

        if hidden_dim is None:

            hidden_dim = in_features * 2

        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim * 2)
        )

    def forward(self, x, *args, **kwargs):
        """
        x - the inpute vector, torch.FloatTensor(..., f)
        """

        # calculate the parameters of the distribution
        mu, log_sigma = self.net(x).chunk(2, -1)

        # sqrt and ensure numerical stability in sigma
        sigma = (log_sigma*0.5).clamp(-5, 4).exp()

        return Normal(mu, sigma)



