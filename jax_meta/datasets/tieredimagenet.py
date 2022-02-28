import numpy as np
import pickle

from jax_meta.datasets.miniimagenet import MiniImagenet


class TieredImagenet(MiniImagenet):
    name = 'tieredimagenet'
    shape = (84, 84, 3)

    def load_data(self):
        if self._data is None:
            arrays, labels2indices = [], {}
            offset = 0
            for data_filename, labels_filename in self.split_filenames:
                with open(data_filename, 'rb') as f:
                    data = np.load(f)
                    arrays.append(data['images'])

                with open(labels_filename, 'rb') as f:
                    data = pickle.load(f)
                    labels = np.asarray(data['labels'])

                    unique_labels = np.unique(labels)
                    labels2indices.update({label: offset + np.where(labels == label)[0]
                        for label in unique_labels})

                    offset += labels.shape[0]
            self._data = np.concatenate(arrays, axis=0)
            self._labels2indices = labels2indices
        return self

    @property
    def split_filenames(self):
        filenames = {
            'train': ('train_images.npz', 'train_labels.pkl'),
            'val': ('val_images.npz', 'val_labels.pkl'),
            'test': ('test_images.npz', 'test_labels.pkl')
        }
        return tuple((self.folder / filenames[split][0],
                      self.folder / filenames[split][1])
            for split in self.splits)

    def _check_integrity(self):
        for data_filename, labels_filename in self.split_filenames:
            if (not data_filename.is_file()) or (not labels_filename.is_file()):
                return False
        return True
