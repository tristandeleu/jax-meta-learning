import numpy as np
import math

from abc import ABC
from pathlib import Path
from numpy.random import default_rng
from collections import namedtuple


Dataset = namedtuple('Dataset', ['inputs', 'targets', 'infos'])


class MetaDataset(ABC):
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
            download=False
        ):
        self.root = Path(root).expanduser()
        self.folder = self.root / self.name
        self.batch_size = batch_size
        self.shots = shots
        self.ways = ways
        self.test_shots = shots if (test_shots is None) else test_shots
        self.size = size

        self.splits = split.split('+')
        assert all(split in ['train', 'val', 'test'] for split in self.splits)

        if download:
            self.download()

        self.seed = seed
        self.reset()

        self._data = None
        self._labels2indices = None
        self._labels = None
        self._num_classes = None

    def reset(self):
        self.rng = default_rng(self.seed)
        self.num_samples = 0
        return self

    @property
    def data(self):
        return self._data

    @property
    def labels2indices(self):
        return self._labels2indices

    @property
    def labels(self):
        if self._labels is None:
            self._labels = sorted(self.labels2indices.keys())
        return self._labels

    @property
    def num_classes(self):
        if self._num_classes is None:
            self._num_classes = len(self.labels)
        return self._num_classes

    def get_indices(self):
        total_shots = self.shots + self.test_shots

        class_indices = np.zeros((self.batch_size, self.ways), dtype=np.int_)
        indices = np.zeros((self.batch_size, self.ways, total_shots), dtype=np.int_)
        targets = np.zeros((self.batch_size, self.ways), dtype=np.int_)

        for idx in range(self.batch_size):
            class_indices[idx] = self.rng.choice(self.num_classes, size=(self.ways,), replace=False)
            targets[idx] = self.rng.permutation(self.ways)

            for way in range(self.ways):
                label = self.labels[class_indices[idx, way]]
                indices[idx, way] = self.rng.choice(self.labels2indices[label], size=(total_shots,), replace=False)

        return class_indices, indices, targets

    def transform(self, data):
        return data

    def __len__(self):
        if self.size is None:
            raise RuntimeError('The dataset has no length because it is infinite.')
        return self.size

    def __iter__(self):
        while (self.size is None) or (self.num_samples < self.size):
            class_indices, indices, targets = self.get_indices()
            data = self.transform(self.data[indices])

            train = Dataset(
                inputs=data[:, :, :self.shots].reshape((self.batch_size, -1) + data.shape[3:]),
                targets=targets.repeat(self.shots, axis=1),
                infos={'labels': class_indices, 'indices': indices[..., :self.shots]}
            )
            test = Dataset(
                inputs=data[:, :, self.shots:].reshape((self.batch_size, -1) + data.shape[3:]),
                targets=targets.repeat(self.test_shots, axis=1),
                infos={'labels': class_indices, 'indices': indices[..., self.shots:]}
            )

            self.num_samples += 1
            yield {'train': train, 'test': test}

    def download(self):
        pass

    @property
    def dummy_input(self):
        return self.transform(np.zeros((1,) + self.data.shape[1:], dtype=np.uint8))

    def flatten(self, num_epochs=1):
        return FlattenDataset(self, num_epochs=num_epochs)


class FlattenDataset:
    def __init__(self, dataset, num_epochs=1):
        self.dataset = dataset
        self.num_epochs = num_epochs
        self.batch_size = dataset.batch_size
        self.size = len(dataset.data)
        self._labels = None
        self.reset()

    @property
    def labels(self):
        if self._labels is None:
            self._labels = np.zeros((self.size,), dtype=np.int_)
            for i, label in enumerate(self.dataset.labels):
                self._labels[self.dataset.labels2indices[label]] = i
        return self._labels

    def reset(self):
        self.dataset.reset()
        self.rng = self.dataset.rng

    def __len__(self):
        return self.num_epochs * math.ceil(self.size / self.batch_size)

    def __iter__(self):
        for _ in range(self.num_epochs):
            indices = self.rng.permutation(self.size)

            for i in range(0, self.size, self.batch_size):
                slice_ = indices[i:i + self.batch_size]
                inputs = self.dataset.transform(self.dataset.data[slice_])
                targets = self.labels[slice_]
                yield inputs, targets
