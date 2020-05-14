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
worker = InferenceWorker('~/agoge/artifacts/dx7-nsp/leaky-burgundy-coati.box', with_data=True)

model = worker.model
data = worker.dataset
loader = data.loaders.test

n_samples = 32
n_latents = 8
loader.batch_sampler.batch_size = n_samples


from uuid import uuid4 as uuid
uuid = lambda: hex(uuid) 
#     self._event.set()


client = jack.Client('DX7Parameteriser')
port = client.midi_outports.register('output')
inport = client.midi_inports.register('input')
event = threading.Event()
fs = None  # sampling rate
offset = 0
from neuralDX7.constants import DX7Single, consume_syx
import torch

name = torch.tensor([i for i in "horns     ".encode('ascii')])

X = torch.zeros(1, 155).long()
X[:,-(len(name)):] = name
X_a = torch.zeros(1, 155).bool()
X_a[:,-(len(name)):] = 1
q = model.features(X, X_a)

X_a[:,-29:] = 1
# X_a = X_a & 0
iter_X = iter(loader)
X_a
syx = list(consume_syx('/home/nintorac/.local/share/DigitalSuburban/Dexed/Cartridges/SynprezFM/SynprezFM_01.syx'))
syx = torch.from_numpy(np.array([list(i.values()) for i in syx]))
# m1, m2 = q.mean[:,0]
# syx_iter = cycle(syx)
def slerp(val, low, high):
    omega = np.arccos(np.clip(np.dot(low/np.linalg.norm(low), high/np.linalg.norm(high)), -1, 1))
    so = np.sin(omega)
    if so == 0:
        return (1.0-val) * low + val * high # L'Hopital's rule/LERP
    return np.sin((1.0-val)*omega) / so * low + np.sin(val*omega) / so * high
x_iter = cycle([*torch.linspace(0, 1, 7)[1:],  *torch.linspace(1, 0, 7)[1:]])
#%%
i=0

mu, std = \
(np.array([ 4.1554513,  4.1125965, -1.9699959,  2.8919716, -6.056072 ,
        -2.407577 , -5.1152377,  2.811712 ], dtype=np.float32),
 np.array([0.3197436 , 0.23426053, 0.25944906, 0.17878139, 0.3019972 ,
        0.34080952, 0.32115632, 0.34698236], dtype=np.float32))

vals = torch.from_numpy(mu + np.linspace(-3, 3, 128)[:,None] * std).float()

controller_map = {}

latent = torch.full((1, 8), 64).long()
patch_no = 0
flow=None
from neuralDX7.utils import mask_parameters
@client.set_process_callback
def process(frames):
    global offset, i
    global msg
    global syx_iter
    global controller_map, patch_no, vals, latent, flow
    port.clear_buffer()
    needs_update = False
    X = syx[[patch_no]]
    a = X_a

    for offset, data in inport.incoming_midi_events():
        msg = mido.parse(bytes(data))

        if msg.type=='note_on':
            # print(msg.__dir__())
            patch_no = msg.note%32
            print(f"patch set to {patch_no}")
            needs_update = True

            a = X_a[[0]]
            flow = model.features(X, a|1)
            q = flow.q_z
            vals = q.mean + torch.linspace(-4, 4, 128)[:,None] * q.stddev 
        
        if msg.type!='control_change':
            continue


        if msg.control not in controller_map:
            if len(controller_map) == 8:
                continue
            print(f"latent {len(controller_map)} set to encoder {msg.control}")
            controller_map[msg.control] = len(controller_map)
        l_i = list(controller_map).index(msg.control)
        print(f'Latent: {latent}')
        latent[:, controller_map[msg.control]] =  msg.value
        needs_update = True
        
        # print("{0}: 0x{1}".format(client.last_frame_time + offset,
        #                           binascii.hexlify(data).decode()))
    # print(time.time()-offset)
    inport.clear_buffer()
    if (needs_update):
        offset = time.time()

        # X = next(iter_X)['x'][[0]]
        # X_d = torch.distributions.Categorical(logits=mask_parameters(torch.zeros(1, 155, 128)))

        # X_a = torch.rand_like(X.float()) < 0.3
        # X_a = torch.ones_like(X).bool()
        # X_a[:,:-10] = 0
        


        # X[~X_a] = X_d.sample()[~X_a]

        # max_to_sample = max((~X_a).sum(-1))
        # # X = X[[0]*1]

        # # for i in tqdm(range(max_to_sample)):

        # logits = model.generate(X, X_a)
        # samples = logits.sample()
            
        # batch_idxs, sample_idx = (~X_a).nonzero().t()

        # X[batch_idxs, sample_idx] = samples[batch_idxs, sample_idx]
        # X_a[batch_idxs, sample_idx] = 1

        # z = slerp(next(x_iter), m1, m2).unsqueeze(-2)[...,[0]*155,:].unsqueeze(0)
        # z = q.mean
        # z = torch.from_numpy(latent).float()
        # print(q.stddev)
        # z = torch.randn_like(z)
        # z = z.mean().unsqueeze
        # z = q.mean# + torch.randn(q.mean.shape) * 0.1 + q.stddev
        # val = torch.from_numpy(vals).float()
        # print(latent)
        # print(time.time())
        z, _ = flow.flow(vals.gather(0, latent))
        # print(time.time())
        # print(a)
        msg = model.generate_z(X, a, z, t=0.001).sample()
        # print(time.time())
        msg[a] = X[a]
        # msg = X if i%2 else msg
        # msg[:,-(len(name)):] = name

        # msg = DX7Single.to_syx([list(next(syx_iter).values())])
        msg = DX7Single.to_syx(msg.numpy().tolist())
        # import mido
        # print([int(i, 2)mido.Message('control_change', control=123).bytes()])
        port.write_midi_event(0, msg.bytes())
        port.write_midi_event(1, mido.Message('control_change', control=123).bytes())
        port.write_midi_event(2, mido.Message('control_change', control=123).bytes())
        port.write_midi_event(3, mido.Message('control_change', control=123).bytes())
        port.write_midi_event(4, mido.Message('control_change', control=123).bytes())

    


@client.set_samplerate_callback
def samplerate(samplerate):
    global fs
    fs = samplerate


@client.set_shutdown_callback
def shutdown(status, reason):
    print('JACK shutdown:', reason, status)
    event.set()

capture_port = 'a2j:Arturia BeatStep [24] (capture): Arturia BeatStep MIDI 1'
playback_port = 'Carla:Dexed:events-in' 

with client:
    # print(client.get_ports())
    offset = time.time()
    # if connect_to:
    port.connect(playback_port)
    inport.connect(capture_port)

    # print('Playing', repr(filename), '... press Ctrl+C to stop')
    try:
        event.wait()
    except KeyboardInterrupt:
        print('\nInterrupted by user')


# %%
