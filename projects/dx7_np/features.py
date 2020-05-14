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
worker = InferenceWorker('/home/nintorac/agoge/artifacts/bluesy-chestnut-forest_0_2020-04-30_11-11-00x02v59be/checkpoint_220/model.box', with_data=True)

model = worker.model
data = worker.dataset
loader = data.loaders.test

n_samples = 32
n_latents = 8
loader.batch_sampler.batch_size = n_samples
features_all = []
features_half = []
for x in map(lambda x: x['x'], tqdm(loader)):
    q = model.features(x, torch.ones_like(x.float()).bool())
    features_all += [(q.mean.numpy(), q.stddev.numpy())]
    # features_half += [model.features(x, torch.rand_like(x.float())>torch.linspace(0, 1, 32).unsqueeze(-1)).mean.numpy()]

# for item in loader:
# %%
