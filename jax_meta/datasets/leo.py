import numpy as np
import os
import pickle

from collections import defaultdict

from jax_meta.datasets.base import MetaDataset
from jax_meta.utils.data import download_url


class LEOMetaDataset(MetaDataset):
    name = 'leo'
    zip_url = 'http://storage.googleapis.com/leo-embeddings/embeddings.zip'
    shape = (640,)

    def __init__(
            self,
            root,
            batch_size,
            shots=5,
            ways=5,
            test_shots=None,
            size=None,
            crop='center',
            split='train',
            seed=0,
            download=False
        ):
        self.crop = crop
        super().__init__(root, batch_size, shots=shots, ways=ways,
            test_shots=test_shots, size=size, split=split, seed=seed,
            download=download)
        self.load_data()

    def load_data(self):
        if self._data is None:
            arrays, labels2indices = [], defaultdict(list)
            offset = 0
            for filename in self.split_filenames:
                with open(filename, 'rb') as f:
                    data = pickle.load(f, encoding='latin')
                    arrays.append(data['embeddings'])

                    for i, key in enumerate(data['keys']):
                        _, class_name, _ = str(key).split('-')
                        labels2indices[class_name].append(i + offset)

                    offset += data['embeddings'].shape[0]
            self._data = np.concatenate(arrays, axis=0)
            self._labels2indices = dict((k, np.asarray(v))
                for (k, v) in labels2indices.items())
        return self

    @property
    def split_filenames(self):
        folder = self.folder / 'embeddings' / self.subname / self.crop
        return tuple(folder / f'{split}_embeddings.pkl' for split in self.splits)

    def _check_integrity(self):
        return all(filename.exists() for filename in self.split_filenames)

    def download(self):
        import zipfile

        if self._check_integrity():
            return

        # Download dataset
        download_url(self.zip_url, self.folder)

        # Extract dataset
        filename = self.folder / os.path.basename(self.zip_url)
        if not filename.with_suffix('').is_dir():
            with zipfile.ZipFile(filename, 'r') as f:
                f.extractall(self.folder)
        filename.unlink(missing_ok=True)


class LEOMiniImagenet(LEOMetaDataset):
    subname = 'miniImageNet'


class LEOTieredImagenet(LEOMetaDataset):
    subname = 'tieredImageNet'
