import torch
from torch import nn

class Attention(nn.Module):



    def __init__(self, n_features, n_hidden, n_heads=8, inf=1e9):
        """
        n_features - number of input features
        n_hidden - hidden dim per head
        n_heads - number of heads
        """

        super().__init__()

        self.QKV = nn.Linear(n_features, n_hidden * 3 * n_heads)
        self.n_heads = n_heads
        self._inf = inf

    @property
    def inf(self):
        # if self.training:
            return self._inf
        # return float('inf')

    def forward(self, X, A):

        batch_size, n_items, _ = X.shape


        QKV = self.QKV(X).reshape(batch_size, n_items, -1, 3, self.n_heads)
        n_dims = len(QKV.shape)
        permuter = torch.arange(n_dims).roll(2)
        Q, K, V = torch.unbind(QKV.permute(3, 4, 0, 1, 2), dim=0)

        qk_t = (Q @ K.transpose(-1, -2)) / (self.n_heads**(1/2))
        qk_t_masked =  qk_t.masked_fill(~A, -self._inf)
        
        Y = qk_t_masked.softmax(-1) @ V

        n_dims = len(Y.shape)
        permuter = torch.arange(n_dims).roll(-1)
        Y = Y.permute(1, 2, 3, 0).flatten(-2, -1)
        return Y



if __name__=="__main__":



    model = Attention(100, 20)
    X = torch.randn(3, 25, 100)
    A = torch.rand(3, 25, 25)>0.5
    Y = model(X, A)
    print(Y.shape)
