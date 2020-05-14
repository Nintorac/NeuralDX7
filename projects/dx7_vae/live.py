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
worker = InferenceWorker('~/agoge/artifacts/dx7-vae/hasty-copper-dogfish_0_2020-05-06_10-46-27o654hmde/checkpoint_204/model.box', with_data=False)
float32='float32'
model = worker.model
# data = worker.dataset
# loader = data.loaders.test

n_samples = 32
n_latents = 8
# loader.batch_sampler.batch_size = n_samples


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
(array([ 5.5626068e-02,  7.9248362e-04, -8.0890575e-04,  1.6684370e-01,
         1.6537485e-01, -6.2455550e-02,  9.4467170e-05, -7.5367272e-02],
       dtype=float32),
 array([0.35453376, 0.3556142 , 0.35896832, 0.341505  , 0.3299536 ,
        0.33990443, 0.3350083 , 0.339214  ], dtype=float32))
vals = torch.from_numpy(mu + np.linspace(-3, 3, 128)[:,None] * std).float()

controller_map = {}

latent = torch.full((1, 8), 64).long()
patch_no = 0

from neuralDX7.utils import mask_parameters
@client.set_process_callback
def process(frames):
    global offset, i
    global msg
    global syx_iter
    global controller_map, patch_no, vals, latent
    port.clear_buffer()
    needs_update = False


    for offset, data in inport.incoming_midi_events():
        msg = mido.parse(bytes(data))

        if msg.type=='note_on':
            port.write_midi_event(0, mido.Message('control_change', control=123).bytes())   
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

        z = vals.gather(0, latent)
        msg = model.generate(z, t=0.001).sample()

        msg = DX7Single.to_syx(msg.numpy().tolist())

        port.write_midi_event(1, msg.bytes())
        mido.write_syx_file('example_single_voice.mid', [msg])
        # port.write_midi_event(1, mido.Message('control_change', control=123).bytes())
        # port.write_midi_event(2, mido.Message('control_change', control=123).bytes())
        # port.write_midi_event(3, mido.Message('control_change', control=123).bytes())
        # port.write_midi_event(4, mido.Message('control_change', control=123).bytes())

    


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
