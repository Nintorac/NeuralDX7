import torch
from torch import nn

class FeedForwardGELU(nn.Module):


    def __init__(self, features, out_features=None, exapnsion_factor=3):

        super().__init__()
        out_features = features if out_features is None else out_features

        # self.net = nn.ModuleList(
        self.net = nn.Sequential(
            # [
            nn.Linear(features, features*exapnsion_factor),
            nn.GELU(),
            nn.Linear(features*exapnsion_factor, out_features)
            # ]
        )

    def forward(self, x):

        return self.net(x)