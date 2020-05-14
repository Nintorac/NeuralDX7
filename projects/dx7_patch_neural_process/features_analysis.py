# %%
from agoge import InferenceWorker
import torch
from tqdm import tqdm
from matplotlib import pyplot as plt
worker = InferenceWorker('/home/nintorac/agoge/artifacts/Worker/messy-firebrick-barracuda_0_2020-04-14_08-59-40ryp93n44/checkpoint_4/model.box', with_data=True)

model = worker.model
data = worker.dataset
loader = data.loaders.test


Xs = []
features = []
for X in tqdm(loader):
    Xs += [X['x']]
    features += [model.features(X['x'])]
features = torch.cat(features).flatten(-2, -1)
X = torch.cat(Xs)
    
#%%
### --------TSNE-----------

import numpy as np
from sklearn.manifold import TSNE
X_embedded = TSNE(n_components=2).fit_transform(features)
X_embedded.shape
#%%
plt.figure(figsize=(20,30))
plt.scatter(*zip(*X_embedded), linewidths=0.1)

# %%
### ---------K-means---------------

from sklearn.cluster import KMeans
from neuralDX7.utils import dx7_bulk_pack
import mido
n_clusters = 8
kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(features)
labels = kmeans.labels_
tasting = []

out_template = '/home/nintorac/.local/share/DigitalSuburban/Dexed/cartridges/neuralDX7/group_{}.syx'

for i in range(n_clusters):
    in_cluster, = (labels == i).nonzero()
    if len(in_cluster) < 32:
        continue

    choices = np.random.choice(in_cluster, 32)
    voices = X[choices]

    patch_message = dx7_bulk_pack(voices)

    mido.write_syx_file(out_template.format(i), [patch_message])



# %%
