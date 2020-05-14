import torch
from torch.nn import functional as F
from importlib import import_module
from torch.optim import AdamW

from agoge import AbstractSolver



class DX7PatchProcess(AbstractSolver):

    def __init__(self, model,
        Optim=AdamW, optim_opts=dict(lr= 1e-4),
        max_beta=0.5,
        **kwargs):

        if isinstance(Optim, str):
            Optim = import_module(Optim)


        self.optim = Optim(params=model.parameters(), **optim_opts)
        self.max_beta = max_beta
        self.model = model

    def loss(self, x, x_hat, x_a):

        valid_predictions = (~x_a).nonzero().t()

        valid_x_hat = x_hat[(*valid_predictions,)]
        valid_x = x[(*valid_predictions,)]

        reconstruction_loss = F.cross_entropy(valid_x_hat, valid_x)
        accuracy = (valid_x_hat.argmax(-1)==valid_x).float().mean()

        return reconstruction_loss, {
            'accuracy': accuracy,
            'reconstruction_loss': reconstruction_loss,
        }
        

    def solve(self, x, **kwargs):
        
        x_hat, x_a  = self.model(x)
        loss, L = self.loss(x, x_hat, x_a)

        if loss != loss:
            raise ValueError('Nan Values detected')

        if self.model.training:

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
        
        return L

    
    def step(self):

        pass


    def state_dict(self):
        
        state_dict = {
            'optim': self.optim.state_dict()
        }

        return state_dict

    def load_state_dict(self, state_dict):
        
        self.optim.load_state_dict(state_dict['optim'])