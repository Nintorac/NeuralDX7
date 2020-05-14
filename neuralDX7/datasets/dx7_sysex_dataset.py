from pathlib import Path
import numpy as np
import torch
from neuralDX7 import DEFAULTS




class DX7SysexDataset():
    

    def __init__(self, data_file='dx7.npy', root=DEFAULTS['ARTIFACTS_ROOT'], data_size=1.):

        assert data_size <= 1

        self.data_size = data_size

        if not isinstance(root, Path):
            root = Path(root).expanduser()

        self.data = np.load(root.joinpath(data_file)) 

    def __getitem__(self, index):

        item = torch.tensor(self.data[index].item()).long()

        return {'X': item}
    
    def __len__(self):
        return int(len(self.data) * self.data_size)


if __name__ == "__main__":
    

    dataset = DX7SysexDataset()

    print([dataset[i] for i in np.random.randint(0, len(dataset)-1, 20)])