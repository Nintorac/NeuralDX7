import torch
from torch import nn

class FeedForwardGELU(nn.Module):
    """
    Simple wrapper for two layer projection with GeLU non linearity

    """

    def __init__(self, features, out_features=None, exapnsion_factor=3):
        """
        features - the number of input features
        out_features - the number of output features, if None copies the input dimension
        expansion_factor - the size of the hidden dimension as a factor of the input features
        """

        super().__init__()
        out_features = features if out_features is None else out_features

        self.net = nn.Sequential(
            nn.Linear(features, features*exapnsion_factor),
            nn.GELU(),
            nn.Linear(features*exapnsion_factor, out_features)
        )

    def forward(self, x):

        return self.net(x)