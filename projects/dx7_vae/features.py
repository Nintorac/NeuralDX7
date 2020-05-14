# %%
from agoge import InferenceWorker
import threading
import torch
import time
import numpy as np
from tqdm import tqdm
import jack
from matplotlib import pyplot as plt
from itertools import cycle
worker = InferenceWorker('~/agoge/artifacts/dx7-vae/hasty-copper-dogfish_0_2020-05-06_10-46-27o654hmde/checkpoint_204/model.box', with_data=True)

model = worker.model
data = worker.dataset
loader = data.loaders.test

n_samples = 32
n_latents = 8
loader.batch_sampler.batch_size = n_samples
features_all = []
features_half = []
for x in map(lambda x: x['X'], tqdm(loader)):
    q = model.features(x)
    features_all += [(q.mean.numpy(), q.stddev.numpy())]
    # features_half += [model.features(x, torch.rand_like(x.float())>torch.linspace(0, 1, 32).unsqueeze(-1)).mean.numpy()]

mus, vars = map(np.concatenate, zip(*features_all))
print(mus.mean(0), vars.mean(0))
# for item in loader:
# %%
from timeit import timeit

with timeit():
    model.generate(q.sample()[[0]])

# %%
from time import time
s_time = time()

for i in range(100):
    model.generate(q.sample()[[0]])

print((time()-s_time)/100)
# %%
