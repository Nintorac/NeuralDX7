# %%
from agoge import InferenceWorker
import threading
import torch
import mido
import time
import numpy as np
from tqdm import tqdm
import jack
from matplotlib import pyplot as plt
from itertools import cycle
from numpy import array
worker = InferenceWorker('hasty-copper-dogfish', 'dx7-vae', with_data=True)
float32='float32'
model = worker.model
# data = worker.dataset
# loader = data.loaders.test

n_samples = 32
n_latents = 8
# loader.batch_sampler.batch_size = n_samples


# randoms = torch.cat([model.generate(torch.randn(2**11, 8)).logits.argmax(-1) for _ in tqdm(range(2**5))])

# %%
from matplotlib import pyplot as plt
import torch

rand = torch.rand(100)
randn = torch.randn(100)
plt.scatter(rand, torch.sigmoid(rand)+0.5)
plt.scatter(randn, torch.sigmoid(randn)-0.5)

# %%
