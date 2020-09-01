from pathlib import Path
import numpy as np
import torch
from neuralDX7 import DEFAULTS




class DX7SysexDataset():
    """
    Pytorch Dataset module to provide access to precprocessed DX7 patch data
    """
    

    def __init__(self, data_file='dx7.npy', root=DEFAULTS['ARTIFACTS_ROOT'], data_size=1.):
        """
        data_file - the name of the prprocessed data
        root - the root directory for data
        data_size - how much of the data is used. good for development
        """

        assert data_size <= 1
        self.data_size = data_size

        # initialise path handler
        if not isinstance(root, Path):
            root = Path(root).expanduser()

        # load data into memory
        self.data = np.load(root.joinpath(data_file)) 

    def __getitem__(self, index):
        
        # turn the data item into a tensor and return
        item = torch.tensor(self.data[index].item()).long()

        return {'X': item}
    
    def __len__(self):
        return int(len(self.data) * self.data_size)


if __name__ == "__main__":
    

    dataset = DX7SysexDataset()

    print([dataset[i] for i in np.random.randint(0, len(dataset)-1, 20)])