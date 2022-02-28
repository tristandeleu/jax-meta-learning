import numpy as np
import os
import pickle

from jax_meta.datasets.base import MetaDataset
from jax_meta.utils.data import copy_dataset_from_repository
import jax_meta.datasets.transforms.functional as F


class MiniImagenet(MetaDataset):
    name = 'miniimagenet'
    shape = (84, 84, 3)

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
        if data_augmentation is None:
            self._data_augmentation = ('train' in self.splits)
        else:
            self._data_augmentation = data_augmentation
        self._mean = np.array([120.39586422, 115.59361427, 104.54012653]) / 255.
        self._std = np.array([70.68188272, 68.27635443, 72.54505529]) / 255.

    def load_data(self):
        if self._data is None:
            arrays, labels2indices = [], {}
            offset = 0
            for filename in self.split_filenames:
                with open(filename, 'rb') as f:
                    data = pickle.load(f, encoding='latin1')
                    arrays.append(data['data'])
                    labels = np.asarray(data['labels'])
                    catname2label = data['catname2label']

                    labels2indices.update({key: offset + np.where(labels == index)[0]
                        for (key, index) in catname2label.items()})
                    offset += data['data'].shape[0]
            self._data = np.concatenate(arrays, axis=0)
            self._labels2indices = labels2indices
        return self

    def transform(self, data):
        if self._data_augmentation:
            data = F.random_horizontal_flip(data, rng=self.rng)
            data = data.astype(np.float32) / 255.
            data = F.color_jitter(data, brightness=0.4, contrast=0.4, saturation=0.4, rng=self.rng)
            data = F.normalize(data, self._mean, self._std)
            data = F.random_crop(data, size=84, padding=8, rng=self.rng)

        else:
            data = data.astype(np.float32) / 255.
            data = F.normalize(data, self._mean, self._std)

        return data

    @property
    def split_filenames(self):
        filenames = {
            'train': 'miniImageNet_category_split_train_phase_train.pickle',
            'val': 'miniImageNet_category_split_val.pickle',
            'test': 'miniImageNet_category_split_test.pickle'
        }
        return tuple(self.folder / filenames[split] for split in self.splits)

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
