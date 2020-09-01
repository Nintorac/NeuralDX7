import torch
from torch.nn import functional as F
from importlib import import_module
from torch.optim import AdamW
from torch.distributions.kl import kl_divergence
from torch.distributions import Normal
from agoge import AbstractSolver

from .utils import sigmoidal_annealing

class DX7VAE(AbstractSolver):
    """
    Solver used to train DX7VAE model
    """

    def __init__(self, model,
        Optim=AdamW, optim_opts=dict(lr= 1e-4),
        max_beta=0.5,
        beta_temp=1e-4,
        **kwargs):

        if isinstance(Optim, str):
            Optim = import_module(Optim)

        self.optim = Optim(params=model.parameters(), **optim_opts)
        self.max_beta = max_beta
        self.model = model

        self.iter = 0
        self.beta_temp = beta_temp

    def loss(self, X, X_hat, flow):
        """
        Computes the VAE loss objective and collects some training statistics

        X - data tensor, torch.LongTensor(batch_size, num_parameters=155)
        X_hat - data tensor, torch.FloatTensor(batch_size, num_parameters=155, max_value=128)
        flow - the namedtuple returned by TriangularSylvesterFlow
        
        for reference, the namedtuple is ('Flow', ('q_z', 'log_det', 'z_0', 'z_k', 'flow'))
        """
    
        p_z_k = Normal(0,1).log_prob(flow.z_k).sum(-1)
        q_z_0 = flow.q_z.log_prob(flow.z_0).sum(-1)
        kl = (q_z_0-p_z_k-flow.log_det).mean() / flow.z_k.shape[-1]

        beta = sigmoidal_annealing(self.iter, self.beta_temp).item()

        reconstruction_loss = F.cross_entropy(X_hat.transpose(-1, -2), X)
        accuracy = (X_hat.argmax(-1)==X).float().mean()

        loss = reconstruction_loss + self.max_beta * beta * kl

        return loss, {
            'accuracy': accuracy,
            'reconstruction_loss': reconstruction_loss,
            'kl': kl,
            'beta': beta,
            'log_det': flow.log_det.mean(),
            'p_z_k': p_z_k.mean(),
            'q_z_0': q_z_0.mean(),
            # 'iter': self.iter // self.
        }

    def solve(self, X, **kwargs):
        """
        Take a gradient step given an input X

        X - data tensor, torch.LongTensor(batch_size, num_parameters=155)
        """
        
        Y = self.model(**X)
        loss, L = self.loss(**X, **Y)

        if loss != loss:
            raise ValueError('Nan Values detected')

        if self.model.training:
            self.iter += 1
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

        return L

    def step(self):

        pass

    def state_dict(self):
        
        state_dict = {
            'optim': self.optim.state_dict(),
            'iter': self.iter
        }

        return state_dict

    def load_state_dict(self, state_dict):
        
        self.optim.load_state_dict(state_dict['optim'])
        self.iter = state_dict['iter']