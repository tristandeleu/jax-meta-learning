import pytest

import numpy as np
from numpy.random import default_rng

import jax_meta.datasets.transforms.functional as F


def test_random_crop():
    # Random data
    rng = default_rng(0)
    data = rng.integers(256, size=(2, 25, 84, 84, 3), dtype=np.uint8)

    # Random crop
    rng = default_rng(1)
    data_F = F.random_crop(data, size=84, padding=8, rng=rng)
    assert data_F.shape == (2, 25, 84, 84, 3)
    assert data_F.dtype == data.dtype

    # Numpy crop
    rng = default_rng(1)  # Same RNG
    data_np = np.zeros_like(data)
    padded = np.pad(data, ((0, 0), (0, 0), (8, 8), (8, 8), (0, 0)))
    x = rng.integers(16, size=(2, 25, 1, 1, 1))
    y = rng.integers(16, size=(2, 25, 1, 1, 1))
    for i in range(2):
        for j in range(25):
            for k in range(84):
                for l in range(84):
                    for c in range(3):
                        row = x[i, j, 0, 0, 0] + k
                        col = y[i, j, 0, 0, 0] + l
                        data_np[i, j, k, l, c] = padded[i, j, row, col, c]
    np.testing.assert_equal(data_F, data_np)


def test_random_horizontal_flip():
    # Random data
    rng = default_rng(0)
    data = rng.integers(256, size=(2, 25, 84, 84, 3), dtype=np.uint8)

    # Random horizontal flip
    rng = default_rng(1)
    data_copy = np.copy(data)
    data_F = F.random_horizontal_flip(data_copy, rng=rng)
    assert data_F.shape == (2, 25, 84, 84, 3)
    np.testing.assert_equal(data_copy, data_F)  # inplace changes

    # Numpy horizontal flip
    rng = default_rng(1)  # Same RNG
    to_flip = (rng.random(size=(2, 25)) < 0.5)
    data_np = np.zeros_like(data)
    for i in range(2):
        for j in range(25):
            if to_flip[i, j]:
                data_np[i, j] = data[i, j, :, ::-1]
            else:
                data_np[i, j] = data[i, j]
    np.testing.assert_equal(data_F, data_np)
