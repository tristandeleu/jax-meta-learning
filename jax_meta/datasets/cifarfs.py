import numpy as np
import os
import pickle

from jax_meta.datasets.fc100 import FC100


class CIFARFS(FC100):
    name = 'cifarfs'
    shape = (32, 32, 3)

    @property
    def split_filenames(self):
        return tuple(self.folder / f'CIFAR_FS_{split}.pickle' for split in self.splits)
