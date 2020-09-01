import torch
from torch import nn

from agoge import AbstractModel
from neuralDX7.models.attention import ResidualAttentionEncoder
from neuralDX7.constants import MAX_VALUE, N_PARAMS
from neuralDX7.utils import mask_parameters


class DX7PatchProcess(AbstractModel):
    """
    EXPERIMENTAL AND UNTESTED

    
    """

    def __init__(self, features, encoder):
        
        super().__init__()

        self.embedder = nn.Embedding(MAX_VALUE, features)
        self.encoder = ResidualAttentionEncoder(**encoder)

        self.logits = nn.Linear(features, MAX_VALUE)

    def forward(self, X):
        # print(X.shape, )

        # generate random masks
        batch_p = torch.rand(X.shape[0]) # decide p value for each item in batch
        item_logits = torch.rand(X.shape) # random value for each param
        X_a = batch_p.unsqueeze(-1) <= item_logits # active params in X
        X_a = X_a.to(self.device)

        A = (~X_a.unsqueeze(-1)) & (X_a.unsqueeze(-2))
        eye = torch.eye(A.shape[-1]).bool().to(self.device) & (~X_a.unsqueeze(-2))
        A = A | eye
        # 1/0

        X = self.embedder(X) * X_a.unsqueeze(-1).float()
        # X = self.embedder(X) 

        X = self.encoder(X, A)
        # X_hat = mask_parameters(self.logits(X))
        X_hat = self.logits(X)
        # print(X_hat.max(), X_hat.min())

        return X_hat, X_a

    @torch.no_grad()
    def features(self, X):

        X_a = torch.ones_like(X).bool()
        A = X_a.unsqueeze(-1) & X_a.unsqueeze(-2)
        X = self.embedder(X)
        # X = self.embedder(X) 

        X = self.encoder(X, A)

        return X

    @torch.no_grad()
    def generate(self, X, X_a):
        
    
        A = (~X_a.unsqueeze(-1)) & (X_a.unsqueeze(-2))
        eye = torch.eye(A.shape[-1]).bool().to(self.device) & (~X_a.unsqueeze(-2))
        A = A | eye

        X = self.embedder(X) * X_a.unsqueeze(-1).float()
        X = self.encoder(X, A)
        X_hat = mask_parameters(self.logits(X))

        X_hat = torch.distributions.Categorical(logits=X_hat)

        return X_hat



if __name__=='__main__':

    layer_features = 100
    n_heads = 4
    N_PARAMS = 8

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



