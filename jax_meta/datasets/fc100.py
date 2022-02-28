import numpy as np
import pickle

from jax_meta.datasets.base import MetaDataset
from jax_meta.utils.data import copy_dataset_from_repository
import jax_meta.datasets.transforms.functional as F


class FC100(MetaDataset):
    name = 'fc100'
    shape = (32, 32, 3)

    def __init__(
            self,
            root,
            batch_size,
            shots=5,
            ways=5,
            test_shots=None,
            size=None,
            split='train',
            seed=0,
            data_augmentation=None,
            download=False
        ):
        super().__init__(root, batch_size, shots=shots, ways=ways,
            test_shots=test_shots, size=size, split=split, seed=seed,
            download=download)
        self.load_data()
        if self.data_augmentation is None:
            self._data_augmentation = ('train' in self.splits)
        else:
            self._data_augmentation = data_augmentation
        self._mean = np.array([129.37731888, 124.10583864, 112.47758569]) / 255.
        self._std = np.array([68.20947949, 65.43124043, 70.45866994]) / 255.

    def load_data(self):
        if self._data is None:
            arrays, labels2indices = [], {}
            offset = 0
            for filename in self.split_filenames:
                with open(filename, 'rb') as f:
                    data = pickle.load(f, encoding='latin1')
                    arrays.append(data['data'])
                    labels = np.asarray(data['labels'])

                    unique_labels = np.unique(labels)
                    labels2indices.update({label: offset + np.where(labels == label)[0]
                        for label in unique_labels})

                    offset += labels.shape[0]
            self._data = np.concatenate(arrays, axis=0)
            self._labels2indices = labels2indices
        return self

    def transform(self, data):
        if self._data_augmentation:
            data = F.random_horizontal_flip(data, rng=self.rng)
            data = data.astype(np.float32) / 255.
            data = F.color_jitter(data, brightness=0.4, contrast=0.4, saturation=0.4, rng=self.rng)
            data = F.normalize(data, self._mean, self._std)
            data = F.random_crop(data, size=32, padding=4, rng=self.rng)

        else:
            data = data.astype(np.float32) / 255.
            data = F.normalize(data, self._mean, self._std)

        return data

    @property
    def split_filenames(self):
        return tuple(self.folder / f'FC100_{split}.pickle' for split in self.splits)

    def _check_integrity(self):
        return all(filename.exists() for filename in self.split_filenames)

    def download(self):
        import zipfile

        if self._check_integrity():
            return

        self.folder.mkdir(exist_ok=True)
        filename = copy_dataset_from_repository(self.name, self.folder)

        with zipfile.ZipFile(filename, 'r') as f:
            f.extractall(self.folder)
        filename.unlink(missing_ok=True)
