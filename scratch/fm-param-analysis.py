#%%
import torch
from torch.utils.data import Subset as DataSubset
from sklearn.model_selection import train_test_split
from fm_param_vae import Net, DX7Dataset, ARTIFACTS_ROOT, VOICE_KEYS
from dx7_constants import VOICE_PARAMETER_RANGES

import numpy as np

dataset = DX7Dataset()
train_idxs, test_idxs = train_test_split(range(len(dataset)), random_state=42)
train_dataset = DataSubset(dataset, train_idxs)
test_dataset = DataSubset(dataset, test_idxs)

model = Net()
model.load_state_dict(torch.load(ARTIFACTS_ROOT.joinpath('fm-param-vae-8.pt')))


# %%
with torch.no_grad():
    data = torch.stack([test_dataset[i] for i in range(len(test_dataset))])
    results, *_ = model(data)
# %%

correct = (results.argmax(-1) == data)
# %%

p_z = torch.distributions.Normal(0, 1)
z = p_z.sample((32, 8))

x_hat = model.generate(z, 0.3)


# %%
