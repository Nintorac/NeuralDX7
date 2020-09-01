import torch
from torch.nn import functional as F
from importlib import import_module
from torch.optim import AdamW
from torch.distributions.kl import kl_divergence

from agoge import AbstractSolver

from .utils import sigmoidal_annealing

class DX7NeuralProcess(AbstractSolver):
    """
    EXPERIMENTAL AND UNTESTED
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

    def loss(self, x, x_hat, x_a, q_context, q_target, z):

        valid_predictions = (~x_a).nonzero().t()

        valid_x_hat = x_hat[(*valid_predictions,)]
        valid_x = x[(*valid_predictions,)]

 
        # kl = kl_divergence(q_target, q_context)#[(*valid_predictions,)]
        # kl = kl_divergence(Normal(torch.zeros_like()), q_context)#[(*valid_predictions,)]
        kl = q_target.log_prob(z) - q_context.log_prob(z)
        kl = kl.sum(-1).mean()
        entropy = q_target.entropy().mean()
        beta = sigmoidal_annealing(self.iter, self.beta_temp).item()

        reconstruction_loss = F.cross_entropy(valid_x_hat, valid_x)
        accuracy = (valid_x_hat.argmax(-1)==valid_x).float().mean()

        loss = reconstruction_loss + 0.25 * beta * kl

        return loss, {
            'accuracy': accuracy,
            'reconstruction_loss': reconstruction_loss,
            'kl': kl,
            'entropy': entropy,
            'beta': beta
        }
        

    def solve(self, x, **kwargs):
        
        x_hat, x_a, q_context, q_target, z  = self.model(x)
        loss, L = self.loss(x, x_hat, x_a, q_context, q_target, z)

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