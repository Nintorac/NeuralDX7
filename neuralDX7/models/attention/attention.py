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

        *input_shape, _ = X.shape

        # calculate the query key and value vectors for all data points
        QKV = self.QKV(X).reshape(*input_shape, -1, 3, self.n_heads)

        # permute the heads and qkv vectors to the first dimensions
        n_dims = len(QKV.shape)
        permuter = torch.arange(n_dims).roll(2)
        Q, K, V = QKV.permute(*permuter)

        # calculate the attention values
        qk_t = (Q @ K.transpose(-1, -2)) / (self.n_heads**(1/2))
        qk_t_masked =  qk_t.masked_fill(~A, -self.inf)
        
        # apply the attention values to the values
        Y = qk_t_masked.softmax(-1) @ V

        # restore heads to the final dimension and flatten (effectively concatenating them)
        n_dims = len(Y.shape)
        permuter = torch.arange(n_dims).roll(-1)
        Y = Y.permute(*permuter).flatten(-2, -1)
        return Y



if __name__=="__main__":



    model = Attention(100, 20)
    X = torch.randn(3, 25, 100)
    A = torch.rand(3, 25, 25)>0.5
    Y = model(X, A)
    print(Y.shape)
