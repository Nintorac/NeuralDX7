from pathlib import Path
import numpy as np
import torch
from neuralDX7 import DEFAULTS
from agoge.lmdb_helper import LMDBDataset
from math import floor


class SingleVoiceLMDBDataset():
    

    def __init__(self, keys_file='unique_voice_keys.npy', data_file='dx7-data.lmdb', root=DEFAULTS['ARTIFACTS_ROOT'], data_size=1.):

        assert data_size <= 1 and data_size > 0

        self.data_size = data_size

        if not isinstance(root, Path):
            root = Path(root).expanduser()

        self.data = LMDBDataset(root.joinpath(data_file))
        self.keys = np.load(root.joinpath(keys_file))

    def __getitem__(self, index):

        key = self.keys[index]
        datapoint = self.data.get(key)
        item = torch.tensor(datapoint['voice_params'].item()).long()

        return {'X': item}
    
    def __len__(self):
        return floor(len(self.keys) * self.data_size)


if __name__ == "__main__":
    

    dataset = SingleVoiceLMDBDataset(root='.')

    # print([dataset[i] for i in np.random.randint(0, len(dataset)-1, 20)])
    print(len(dataset))