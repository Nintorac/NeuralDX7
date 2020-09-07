from itertools import chain
from pathlib import Path
import numpy as np
import torch
from neuralDX7 import DEFAULTS
from agoge.lmdb_helper import LMDBDataset
from math import floor
from random import choices
from pronounceable import generate_word



class VoiceMixingLMDBDataset():
    """
    Dynamically creates random patches as combinations of known
    good configurations for each.

    effective dataset size is therefore 
        (len(osc_param) choose 6) + (len(global_param) choose 1)

    i.e a really large number of unique examples. for this reason it is 
    not feasible to go through the whole datset and thus an artifical dateset
    size is supplyable at init
    
    This setup makes it a little hard to do a proper train test split since 
    for example the current dataset has 56604 unique oscillator configurations
    and 9072 unique global configurations giving a total ~4e29 unique parameter
    sets. with a batch size of 32 and 20 iterations/second this would take 
    1.5e9 the age of the universe to compute. or about the same time as it would 
    chunk of bismuth-209 to degrade half of its mass.
    """

    def __init__(self, 
                data_file='dx7-split-oscillator-global.lmdb', 
                root=DEFAULTS['ARTIFACTS_ROOT'], dataset_len=80000):

        self.dataset_len = dataset_len

        if not isinstance(root, Path):
            root = Path(root).expanduser()

        self.data_file = LMDBDataset(root.joinpath(data_file), max_dbs=2)

        self.osc_keys = sorted(self.data_file.keys(db='oscillator'))
        self.global_keys = sorted(self.data_file.keys(db='global'))


    def __getitem__(self, index):

        osc_choices = choices(self.osc_keys, k=6)
        osc_params = [self.data_file.get(key, db='oscillator') for key in osc_choices]

        global_choice = choices(self.global_keys, k=1)
        global_params = self.data_file.get(global_choice, db='global')

        name = [ord(i) for i in generate_word().ljust(10, " ")]

        item = list(chain(*osc_params, *global_params, name))
        item = np.array(item, dtype='int32')

        return {'X': torch.tensor(item).long()}
    
    def __len__(self):
        return self.dataset_len


if __name__ == "__main__":
    

    dataset = VoiceMixingLMDBDataset()
    dataset[0]
    