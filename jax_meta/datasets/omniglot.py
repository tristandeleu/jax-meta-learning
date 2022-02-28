import numpy as np
import json
import concurrent.futures

from glob import glob
from PIL import Image
from collections import defaultdict

import jax_meta.datasets.transforms.functional as F
from jax_meta.datasets.base import MetaDataset
from jax_meta.utils.data import download_url, get_asset


class Omniglot(MetaDataset):
    name = 'omniglot'
    url = 'https://raw.githubusercontent.com/brendenlake/omniglot/master/python'
    filenames = ['images_background', 'images_evaluation']
    shape = (28, 28, 1)

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
        super().__init__(root, batch_size, shots=shots, ways=ways,
            test_shots=test_shots, size=size, split=split, seed=seed,
            download=download)
        self.load_data()

    def load_data(self):
        if self._data is None:
            arrays, labels2indices = [], {}
            offset = 0
            for data_filename, labels_filename in self.split_filenames:
                with open(data_filename, 'rb') as f:
                    data = np.load(f)
                    arrays.append(data)

                with open(labels_filename, 'r') as f:
                    labels = json.load(f)

                    labels2indices.update({label: offset + np.array(indices)
                        for label, indices in labels.items()})

                    offset += data.shape[0]
            arrays = np.concatenate(arrays, axis=0)

            # Add class augmentations
            data, labels2indices_aug, num_samples = [], {}, arrays.shape[0]
            for k in [0, 1, 2, 3]:  # Rotation by 0, 90, 180, 270
                data.append(np.rot90(arrays, k=k, axes=(1, 2)))
                labels2indices_aug.update({
                    f'{key}_{k * 90}': values + num_samples * k
                    for (key, values) in labels2indices.items()
                })

            self._data = np.concatenate(data, axis=0)
            self._labels2indices = labels2indices_aug
        return self

    def transform(self, data):
        return 1. - data.astype(np.float32) / 255.

    @property
    def split_filenames(self):
        filenames = {
            'train': ('train_images.npy', 'train_labels.json'),
            'val': ('val_images.npy', 'val_labels.json'),
            'test': ('test_images.npy', 'test_labels.json')
        }
        return tuple((self.folder / filenames[split][0],
                      self.folder / filenames[split][1])
            for split in self.splits)

    def _check_integrity(self):
        for data_filename, labels_filename in self.split_filenames:
            if (not data_filename.is_file()) or (not labels_filename.is_file()):
                return False
        return True

    def download(self, max_workers=8):
        import zipfile
        import shutil

        if self._check_integrity():
            return

        for filename in self.filenames:
            zip_filename = f'{filename}.zip'
            filename = self.folder / zip_filename

            if filename.is_file():
                continue

            download_url(f'{self.url}/{zip_filename}', self.folder, zip_filename)
            with zipfile.ZipFile(filename, 'r') as f:
                f.extractall(self.folder)

        def load_image(filename, size=28):
            image = Image.open(filename, mode='r').convert('L')
            return image.resize((size, size), Image.BILINEAR)

        for split in ['train', 'val', 'test']:
            data = get_asset(self.name, f'{split}.json', dtype='json')
            characters = []
            for folder in ['background', 'evaluation']:
                characters.extend([
                    (f'{alphabet}/{character}', filename)
                    for (alphabet, characters) in data[folder].items()
                    for character in characters
                    for filename in glob(str(self.folder / f'images_{folder}' / alphabet / character / '*.png'))
                ])
            characters = sorted(characters)

            images = np.zeros((len(characters),) + self.shape, dtype=np.uint8)
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                filenames = [filename for _, filename in characters]
                for i, image in enumerate(executor.map(load_image, filenames)):
                    images[i, ..., 0] = image

            labels = defaultdict(list)
            for i, (label, _) in enumerate(characters):
                labels[label].append(i)

            with open(self.folder / f'{split}_images.npy', 'wb') as f:
                np.save(f, images)

            with open(self.folder / f'{split}_labels.json', 'w') as f:
                json.dump(labels, f)

        for filename in self.filenames:
            filename = self.folder / f'{filename}.zip'
            shutil.rmtree(filename.with_suffix(''))
            filename.unlink(missing_ok=True)
