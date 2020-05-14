# %%
from agoge import InferenceWorker
import torch
from tqdm import tqdm
from matplotlib import pyplot as plt
worker = InferenceWorker('/home/nintorac/agoge/artifacts/Worker/messy-firebrick-barracuda_0_2020-04-14_08-59-40ryp93n44/checkpoint_4/model.box', with_data=True)

model = worker.model
data = worker.dataset
loader = data.loaders.train

n_samples = 32

loader.batch_sampler.batch_size = n_samples
# %%
# batch = next(iter(loader))['x']

# X_a = torch.rand_like(batch.float()) > torch.linspace(0, 1, n_samples).unsqueeze(-1)

# logits = model.generate(batch, X_a)

# # %%
# from matplotlib import pyplot as plt
# plt.imshow(X_a)

# # %%
# plt.scatter(torch.arange(n_samples), logits.log_prob(batch).mean(-1))

#     # %%
# plt.imshow(logits.log_prob(batch))

# %%

from itertools import count 
from neuralDX7.utils import dx7_bulk_pack, mask_parameters
import mido
iter_X = iter(loader)
for n in range(10):
    X = next(iter_X)['x']
    # syx = dx7_bulk_pack(X.numpy().tolist())
    # mido.write_syx_file('/home/nintorac/.local/share/DigitalSuburban/Dexed/Cartridges/neuralDX7/OG.syx', [syx])

    X_d = torch.distributions.Categorical(logits=mask_parameters(torch.zeros(32, 155, 128)))

    X_a = torch.rand_like(X.float()) < 0.3
    X_a = torch.ones_like(X).bool()
    X_a[:,:-10] = 0

    X = X[[0]*32]
    X[~X_a] = X_d.sample()[~X_a]

    max_to_sample = max((~X_a).sum(-1))

    for i in tqdm(range(max_to_sample)):

        logits = model.generate(X, X_a)
        samples = logits.sample()

        has_unsampled = ~X_a.all(-1)

        sample_idx = (torch.rand_like(X.float()) * (~X_a).float()).argmax(-1)[has_unsampled]
        batch_idxs = torch.arange(X.shape[0])[has_unsampled]


        X[batch_idxs, sample_idx] = samples[batch_idxs, sample_idx]
        X_a[batch_idxs, sample_idx] = 1
        # X_a = X_a | new_mask

        if X_a.all():
            break

    syx = dx7_bulk_pack(X.numpy().tolist())
    mido.write_syx_file(f'/home/nintorac/.local/share/DigitalSuburban/Dexed/Cartridges/neuralDX7/gen_{n}.syx', [syx])

# # %%
# from neuralDX7.constants import voice_struct, VOICE_KEYS, checksum
# def dx7_bulk_pack(voices):

#     HEADER = int('0x43', 0), int('0x00', 0), int('0x09', 0), int('0x20', 0), int('0x00', 0)
#     assert len(voices)==32
#     voices_bytes = bytes()
#     for voice in voices:
#         voice_bytes = voice_struct.pack(dict(zip(VOICE_KEYS, voice)))
#         voices_bytes += voice_bytes
    
    
#     patch_checksum = [checksum(voices_bytes)]

# #     data = bytes(HEADER) + voices_bytes + bytes(patch_checksum)

# #     return mido.Message('sysex', data=data)

# # %%

# from neuralDX7.constants import VOICE_KEYS, MAX_VALUE, VOICE_PARAMETER_RANGES
# def mask_parameters(x, voice_keys=VOICE_KEYS, inf=1e9):
#     device = x.device
#     mask_item_f = lambda x: torch.arange(MAX_VALUE).to(device) > max(x) 
#     mapper = map(mask_item_f, map(VOICE_PARAMETER_RANGES.get, voice_keys))

#     mask = torch.stack(list(mapper))
    
#     return torch.masked_fill(x, mask, -inf)

# plt.imshow(mask_parameters(torch.randn(10, 155, 128))[0])
# # %%


# %%
