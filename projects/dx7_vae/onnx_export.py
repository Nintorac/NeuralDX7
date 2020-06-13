# %%
from agoge import InferenceWorker
from pathlib import Path
import torch
import numpy as np
from numpy import array

# worker = InferenceWorker('hasty-copper-dogfish', 'dx7-vae', with_data=False)
float32='float32'
# model = worker.model


mu, std = \
(array([ 5.5626068e-02,  7.9248362e-04, -8.0890575e-04,  1.6684370e-01,
         1.6537485e-01, -6.2455550e-02,  9.4467170e-05, -7.5367272e-02],
       dtype=float32),
 array([0.35453376, 0.3556142 , 0.35896832, 0.341505  , 0.3299536 ,
        0.33990443, 0.3350083 , 0.339214  ], dtype=float32))
vals = torch.from_numpy(mu + np.linspace(-3, 3, 128)[:,None] * std).float()

controller_map = {}

# latent = torch.full((1, 8), 64).long()
# l_i = list(controller_map).index(msg.control)
# latent[:, controller_map[msg.control]] =  msg.value
# z = vals.gather(0, latent)

class Model(torch.nn.Module):

       def __init__(self):
              super().__init__()
              worker = InferenceWorker('hasty-copper-dogfish', 'dx7-vae', with_data=False)
              self.model = worker.model

              self.mu, self.std = \
                     (
                            torch.nn.Parameter(torch.from_numpy(array([ 5.5626068e-02,  7.9248362e-04, -8.0890575e-04,  1.6684370e-01,
                            1.6537485e-01, -6.2455550e-02,  9.4467170e-05, -7.5367272e-02],
                            dtype=float32))),
                            torch.nn.Parameter(torch.from_numpy(array([0.35453376, 0.3556142 , 0.35896832, 0.341505  , 0.3299536 ,
                            0.33990443, 0.3350083 , 0.339214  ], dtype=float32)))
                     )
       
       
       def forward(self, z, t):
              """
              z in in the range 0, 1
              """
              print(z)
              z = (z-0.5) * 2
              z = self.mu + z * self.std 
              print(z)
              logits = self.model.generate(z, t)

              # sample = logits.sample()

              return torch.softmax(logits, dim=-1)


model = Model()
sm = torch.jit.script(model)
# sm.save('Model.jit')
torch.jit.save(sm, 'Model.jit')

# Export the modelsp
# torch_out = torch.onnx._export(
#             model,             # model being run
#             (torch.ones(1, 8)*0.5, torch.ones(1)),                       # model input (or a tuple for multiple inputs)
#             Path("~/agoge/dx7-vae-hasty-copper-dogfish.onnx").expanduser().as_posix(), # where to save the model (can be a file or file-like object)
#             export_params=True,
#        #      verbose=True
# )  



# %%
