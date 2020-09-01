# FM Synth Parameter Generator

Random machine learning experiments related to the classic Yamaha DX7

## Dexed

Dexed is a linux open source DX7 emulator and was used heavily in the testing of this project

## SYX

DX7 and it's similar instruments are programmable bt

format [found here](https://github.com/asb2m10/dexed/tree/master/Documentation
) under `sysexformat.txt`

## Dataset

Big thanks to Bobby Blues for collecting [these](http://bobbyblues.recup.ch/yamaha_dx7/dx7_patches.html) DX7 patches. This was the only data source

# Directory structure
```
neuralDX7/
├── constants.py
├── datasets                    # modules to interface with preprocessed datasets 
│   ├── dx7_sysex_dataset.py
│   └── __init__.py
├── __init__.py
├── models
│   ├── attention               # modules implementing transformer stack based on Attention Is All You Need
│   │   ├── attention_encoder.py
│   │   ├── attention_layer.py
│   │   ├── attention.py
│   │   ├── conditional_attention_encoder.py
│   │   └── __init__.py
│   ├── general
│   │   ├── gelu_ff.py          # two layer non linear layer using GeLU non-linearity 
│   │   └── __init__.py
│   ├── __init__.py
│   ├── stochastic_nodes        # layers implementing stochastic transformations
│   │   ├── __init__.py
│   │   ├── normal.py
│   │   └── triangular_sylvester.py
│   ├── dx7_cnp.py              # experimental modules
│   ├── dx7_np.py               # experimental modules
│   ├── dx7_nsp.py              # experimental modules
│   ├── dx7_vae.py              # working model used in production of thisdx7cartdoesnotexist.com
│   └── utils.py
├── solvers
│   ├── dx7_np.py               # experimental modules
│   ├── dx7_nsp.py              # experimental modules
│   ├── dx7_patch_process.py    # experimental modules
│   ├── dx7_vae.py              # working model used in production of thisdx7cartdoesnotexist.com
│   ├── __init__.py
│   └── utils.py
└── utils.py
```


# thisdx7cartdoesnotexist.com

If you've found your way here through [thisdx7cartdoesnotexist.com](https://www.thisdx7cartdoesnotexist.com/) then this section will give an overview of how that site generates patches.


The model itself is defined under `NeuralDX7/models/dx7_vae.py`, it is a simple VAE with triangular sylvester flows implemented with attention layers over the parameters of the DX7. 

The training code can be found in `NeuralDX7/solvers/dx7_vae.py` and is a fairly standard VAE+Flow optimisation setup 

Finally the training script itself, as well as various other scripts used to perform various functions with the trained model can be found under `projects/dx7_vae`. The scripts in here do the following:

`duplicate_test.py` - samples randomly from the prior and calculates the number of identical patches. This was found to be around 99.9% unique

`evaluate.py` - was a script designed to create a single cartridge and was used during development to ensure the model was outputting valid parameter configurations.

`experiment.py` - contains the code to run the experiment. If you want to train your own version then start here.

`features.py` - simple feature extractor for the dataset, used to calculate the model posterior.

`interpolate.py` - takes two samples from the dataset and produces a cartridge that moves between the two in latent space over 32 steps

`live.py` - uses jackd to provide a realtime interface to the model. This is really fun to play with as it lets you hook up a midi controller to control the latent variables of the model and update the parameters of your FM Synthesizer in real time. 

To use it you will need jackd installed, add both your controller and FM synthsizer to the jack graph, update the names of the controller and fm synth in the `live.py` script and then run. Each of the first 8 midi cc controls recieved will be mapped to a latent dimension of the model.


```
import torch
import mido
from agoge import InferenceWorker
from neuralDX7.utils import dx7_bulk_pack

# load model (weights should download automatically)
model = InferenceWorker('hasty-copper-dogfish', 'dx7-vae', with_data=False).model

# sample latent from prior N(0,1)
z = torch.randn(32, 8)

# decode samples to logits
p_x = model.generate(z)

# sample
sample = p_x.logits.argmax(-1)

# convert pytorch tensors to syx
msg = dx7_bulk_pack(sample.numpy().tolist())

mido.write_syx_file('path/to/save.syx', [msg])
```