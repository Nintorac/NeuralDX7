#%%
import os
os.environ['MLFLOW_TRACKING_URI'] = 'http://localhost:9001'
import torch
torch.randn(10,10).cuda()
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from pathlib import Path
from torch.utils.data import Subset as DataSubset
from ray import tune
# from sklearn.model_selection import train_test_split
import numpy as np
from dx7_constants import VOICE_PARAMETER_RANGES, ARTIFACTS_ROOT, VOICE_KEYS

from ray.tune.schedulers import HyperBandForBOHB
from ray.tune.suggest.bohb import TuneBOHB
import ConfigSpace as CS

from agoge import AbstractModel, AbstractSolver, Worker
from agoge.utils import uuid, trial_name_creator, experiment_name_creator, get_logger
from itertools import starmap

import numpy as np
N_PARAMS = len(VOICE_PARAMETER_RANGES)
MAX_VALUE = max([max(i) for i in VOICE_PARAMETER_RANGES.values()]) + 1
#%%

# class DataHandler()
#     def __init__(self, data_file, root=ARTIFACTS_ROOT):

#         if not isinstance(root, Path):
#             root = Path(root).expanduser()

#         data = np.load(ARTIFACTS_ROOT.joinpath(patch_file))



class DX7Dataset():
    

    def __init__(self, data_file='dx7.npy', root=ARTIFACTS_ROOT, data_size=1.):

        assert data_size < 1

        self.data_size = data_size

        if not isinstance(root, Path):
            root = Path(root).expanduser()

        self.data = np.load(root.joinpath(data_file)) 

    def __getitem__(self, index):

        item = torch.tensor(self.data[index].item()).long()

        return {'x': item}
    
    def __len__(self):
        return int(len(self.data) * self.data_size)


#%%

#%%
class DX7RecurrentVAE(AbstractModel):
    def __init__(self, latent_dim=8, n_params=N_PARAMS, max_value=MAX_VALUE, hidden_dim=128, params_ordering=None):
        super().__init__()

        self.n_params = n_params
        self.max_value = max_value

        self.embedder = nn.Embedding(max_value, hidden_dim)

        self.enc = nn.ModuleList(
            [nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional=True),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional=True),]
        )

        self.q_z = nn.Linear(hidden_dim, 2*latent_dim)
        self.z2x = nn.Linear(hidden_dim+latent_dim, hidden_dim)
        self.logits = nn.Linear(hidden_dim, max_value)

        self.dec = nn.ModuleList(
            [nn.LSTM(hidden_dim, hidden_dim, batch_first=True),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.LSTM(hidden_dim, hidden_dim, batch_first=True),
            ]
        )

        self.ordering = params_ordering
        if params_ordering is not None:
            assert len(params_ordering) == len(VOICE_KEYS), 'more or less params than expected'
            self.ordering = np.argsort(params_ordering)
            self.reverse_ordering = np.argsort(self.ordering)


        self.register_buffer('mask', self.generate_mask(self.ordering))


    def network(self, x, network):

        lstm, gelu, drop, lstm2 = network

        x_1, (h_1, _) = lstm(x)
        if lstm.bidirectional == True:
            x_1 = h_1.mean(0)
        x_1 = drop(gelu(x_1))

        x_2, (h_2, _) = lstm2(x)

        if lstm2.bidirectional == True:
            x_2 = h_2.mean(0)
            x = torch.ones_like(x_2)

        x_2 = drop(x_2)

        x = x_1 * x + x_2

        return x

    @staticmethod
    def generate_mask(ordering=None):
        """
        ordering the index ordering of the parameters based on the index in the dx7_constants.VOICE_KEYS
        """
        
        mask_item_f = lambda x: torch.arange(MAX_VALUE) <= max(x) 
        mapper = map(mask_item_f, map(VOICE_PARAMETER_RANGES.get, VOICE_KEYS))

        mask = torch.stack(list(mapper))

        if ordering is not None:
            return mask[ordering]
        return mask

    def forward(self, x):
        if self.ordering is not None:
            x = x[:, self.ordering]

        x = self.embedder(x)
        theta_z = self.network(x, self.enc)

        q_z_mu, q_z_std = self.q_z(theta_z).chunk(2, -1)

        q_z = torch.distributions.Normal(q_z_mu, (0.5*q_z_std.clamp(-5, 3)).exp())

        z = q_z.sample()
        z_in = z.unsqueeze(-2) + torch.zeros_like(x[...,0]).unsqueeze(-1)

        # x_endcut = x
        x_prepad = torch.cat([torch.zeros_like(x[:,[0]]), x], dim=-2)
        x_endcut = x_prepad[:,:-1]
        
        x_dec_in = torch.cat([x_endcut, z_in], dim=-1)
        x_dec_in = self.z2x(x_dec_in)
        x_hat = self.network(x_dec_in, self.dec)

        x_hat = self.logits(x_hat)

        x_hat = torch.masked_fill(x_hat, ~self.mask, -1e9)

        if self.ordering is not None:
            x_hat = x_hat[:, self.reverse_ordering]

        return x_hat, q_z, z

    def generate(self, z, t=1.):

        x_hat = self.dec(z)
        x_hat = x_hat.reshape(-1, self.n_params, self.max_value)
        x_hat = torch.masked_fill(x_hat, ~self.mask, -float('inf'))

        x_hat = torch.distributions.Categorical(logits=x_hat / t)

        return x_hat
#%%

class DX7RecurrentVAESolver(AbstractSolver):

    def __init__(self, model,
        Optim=AdamW, optim_opts=dict(lr= 1e-4),
        max_beta=0.5,
        **kwargs):

        if isinstance(Optim, str):
            Optim = import_module(Optim)


        self.optim = Optim(params=model.parameters(), **optim_opts)
        self.schedule = self.scheduler()
        self.max_beta = max_beta
        self.model = model
        self._beta = self.schedule()

    @property
    def beta(self):

        return self._beta * self.max_beta

    @staticmethod
    def scheduler():
        n_steps  = 0
        beta_steps = 37400

        def schedule():
            nonlocal n_steps
            n_steps += 1

            step = (n_steps)/beta_steps
            step = min(1, step)

            return 0.5 * (1 + np.sin((step*np.pi)-(np.pi/2)))
        return schedule
      
    def loss(self, x, x_hat, q_z, z):

        reconstruction_loss = F.cross_entropy(x_hat.transpose(-1,-2), x)

        p_z = torch.distributions.Normal(0, 1)

        log_q_z = q_z.log_prob(z)
        log_p_z = p_z.log_prob(z)

        kl = (log_q_z - log_p_z).mean()

        kl_tempered = kl * self.beta
        
        loss = reconstruction_loss + kl_tempered


        accuracy = (x_hat.argmax(-1)==x).float().mean()

        return loss, {
            'log_q_z': log_q_z.mean(),
            'log_p_z': log_p_z.mean(),
            'kl': kl,
            'reconstruction_loss': reconstruction_loss,
            'beta': self.beta,
            'accuracy': accuracy
        }
        

    def solve(self, x, **kwargs):
        
        x_hat, q_z, z  = self.model(x)
        loss, L = self.loss(x, x_hat, q_z, z)

        if loss != loss:
            raise ValueError('Nan Values detected')

        if self.model.training:

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            self._beta = self.schedule()
        
        return L

    
    def step(self):

        pass


    def state_dict(self):
        
        state_dict = {
            'optim': self.optim.state_dict()
        }

        return state_dict

    def load_state_dict(self, state_dict):
        
        load_component = lambda component, state: getattr(self, component).load_state_dict(state)
        list(starmap(load_component, state_dict.items()))



def config(experiment_name, trial_name, batch_size=16, **kwargs):
    
    voice_params = {key.split('..')[-1]: value for key, value in kwargs.items() if 'VOICE..' in key}
    params_ordering = list(map(voice_params.get, VOICE_KEYS))


    data_handler = {
        'Dataset': DX7Dataset,
        'dataset_opts': {
            'data_size': 0.2
        },
        'loader_opts': {
            'batch_size': batch_size,
        },
    }

    model = {
        'Model': DX7RecurrentVAE,
        'params_ordering': params_ordering
        # 'conv1': (1, 32, 3, 1)
    }

    solver = {
        'Solver': DX7RecurrentVAESolver
    }

    tracker = {
        'metrics': ['reconstruction_loss', 'log_q_z', 'log_p_z', 'kl', 'beta', 'accuracy'],
        'experiment_name': experiment_name,
        'trial_name': trial_name
    }

    return {
        'data_handler': data_handler,
        'model': model,
        'solver': solver,
        'tracker': tracker,
    }

from mlflow.tracking import MlflowClient
if __name__=='__main__':
    # from ray import ray
    import sys
    postfix = sys.argv[1]
    # ray.init()
    # from ray.tune.utils import validate_save_restore
    # validate_save_restore(Worker)
    # client = MlflowClient(tracking_uri='localhost:5000')
    experiment_name = f'dx7-vae-{postfix}'#+experiment_name_creator()
    # experiment_id = client.create_experiment(experiment_name)


    experiment_metrics = dict(metric="loss/accuracy", mode="max")

    config_space = CS.ConfigurationSpace()
    [config_space.add_hyperparameter(
        CS.UniformFloatHyperparameter(f'VOICE..{key}', lower=0., upper=1)
    ) for key in VOICE_KEYS]
    bohb_hyperband = HyperBandForBOHB(
        time_attr="training_iteration", max_t=16, **experiment_metrics)
    bohb_search = TuneBOHB(
        config_space, max_concurrent=1, **experiment_metrics)


    tune.run(Worker, 
    config={
        'config_generator': config,
        'experiment_name': experiment_name,
        'points_per_epoch': 2
    },
    trial_name_creator=trial_name_creator,
    resources_per_trial={
        'gpu': 1
    },
    checkpoint_freq=2,
    checkpoint_at_end=True,
    keep_checkpoints_num=1,
    search_alg=bohb_search, 
    scheduler=bohb_hyperband,
    num_samples=4096,
    verbose=0,
    local_dir='~/ray_results'
    # webui_host='127.0.0.1' ## supresses an error
        # stop={'loss/loss': 0}
    )
# points_per_epoch
# %%

# #################################################################################
  

# if __name__=="__main__":
#     # Training settings
#     use_cuda = True
#     batch_size = 32
#     lr = 1e-4
#     gamma = 1.
#     epochs = 100
#     beta = 0.5
#     beta_steps = 37400
    
#     schedule = scheduler()
#     device = torch.device("cuda" if use_cuda else "cpu")

#     kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    
#     dataset = DX7Dataset()
#     train_idxs, test_idxs = train_test_split(range(len(dataset)), random_state=42)
#     train_dataset = DataSubset(dataset, train_idxs)
#     test_dataset = DataSubset(dataset, test_idxs)

#     train_loader = torch.utils.data.DataLoader(
#         train_dataset,
#         batch_size=batch_size, shuffle=True, **kwargs)
#     test_loader = torch.utils.data.DataLoader(
#         test_dataset,
#         batch_size=batch_size, shuffle=True, **kwargs)

#     model = Net().to(device)
#     optimizer = optim.AdamW(model.parameters(), lr=lr)

#     scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
#     for epoch in range(1, epochs + 1):
#         train(model, device, train_loader, optimizer, epoch)
#         test(model, device, test_loader)
#         # scheduler.step()

#     # if args.save_model:
#     #     torch.save(model.state_dict(), "mnist_cnn.pt")

#     torch.save(model.state_dict(), ARTIFACTS_ROOT.joinpath('fm-param-vae-8.pt'))
# # if __name__ == '__main__':
# #     main()

# # %%


# # %%


# # %%


# # %%
