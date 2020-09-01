import torch
from torch.nn import functional as F
from importlib import import_module
from torch.optim import AdamW
from torch.distributions.kl import kl_divergence

from agoge import AbstractSolver

from .utils import sigmoidal_annealing

class DX7NeuralSylvesterProcess(AbstractSolver):
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

    def loss(self, X, X_hat, X_a, flow_context, flow_target):

        valid_predictions = (~X_a).nonzero().t()

        valid_x_hat = X_hat[(*valid_predictions,)]
        valid_x = X[(*valid_predictions,)]

        p_z = flow_target.q_z.log_prob(flow_context.z_k).sum(-1)
        q_z = flow_context.q_z.log_prob(flow_context.z_0).sum(-1)
        kl = (q_z-p_z-flow_context.log_det).mean() / flow_context.z_k.shape[-1]
        beta = sigmoidal_annealing(self.iter, self.beta_temp).item()

        reconstruction_loss = F.cross_entropy(valid_x_hat, valid_x)
        accuracy = (valid_x_hat.argmax(-1)==valid_x).float().mean()

        loss = reconstruction_loss + self.max_beta * beta * kl

        return loss, {
            'accuracy': accuracy,
            'reconstruction_loss': reconstruction_loss,
            'kl': kl,
            'beta': beta,
            'q_log_det': flow_context.log_det.mean(),
            # 'p_log_det': flow_target.log_det.mean(),
            'q_z': q_z.mean(),
            'p_z': p_z.mean()
        }
        

    def solve(self, X, **kwargs):
        
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