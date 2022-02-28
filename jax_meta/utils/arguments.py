import os
import enum

from typing import Optional
from dataclasses import dataclass
from simple_parsing.helpers import Serializable, encode

from jax_meta import datasets


class Dataset(enum.Enum):
    omniglot = 'omniglot'
    miniimagenet = 'miniimagenet'
    tieredimagenet = 'tieredimagenet'
    cifarfs = 'cifarfs'
    fc100 = 'fc100'

    @property
    def _cls(self):
        classes = {
            Dataset.omniglot: datasets.Omniglot,
            Dataset.miniimagenet: datasets.MiniImagenet,
            Dataset.tieredimagenet: datasets.TieredImagenet,
            Dataset.cifarfs: datasets.CIFARFS,
            Dataset.fc100: datasets.FC100
        }
        if self not in classes:
            raise ValueError(f'Unknown dataset: `{self}`')

        return classes[self]


@encode.register(Dataset)
def encode_dataset(dataset):
    return dataset.value


@dataclass
class DataArguments(Serializable):
    # data folder
    folder: Optional[str]

    # dataset name
    dataset: Dataset = Dataset.miniimagenet

    # number of classes per task (N in "N-way", default: %(default)s)
    ways: int = 5

    # number of training examples per class (k in "k-shot", default: %(default)s)
    shots: int = 5

    # number of test examples per class
    test_shots: int = 15

    # random seed
    seed: int = 1

    def __post_init__(self):
        if self.folder is None:
            if os.getenv('SLURM_TMPDIR') is not None:
                self.folder = os.path.join(os.getenv('SLURM_TMPDIR'), 'data')
            else:
                raise ValueError(f'Invalid value of `folder`: {self.folder}. '
                    '`folder` must be a valid folder.')
        os.makedirs(self.folder, exist_ok=True)

        if self.test_shots <= 0:
            self.test_shots = self.shots
