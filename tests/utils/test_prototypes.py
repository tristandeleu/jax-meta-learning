import pytest

import numpy as np
import jax.numpy as jnp

from jax import random, vmap

from jax_meta.utils.prototypes import get_num_samples, get_prototypes


@pytest.mark.parametrize('dtype', [None, jnp.int32, jnp.float32])
def test_get_num_samples(dtype):
    num_classes = 3
    targets = jnp.array([1, 0, 2, 1, 0], dtype=jnp.int32)

    num_samples = get_num_samples(targets, num_classes, dtype=dtype)

    assert num_samples.shape == (num_classes,)
    if dtype is not None:
        assert num_samples.dtype == dtype

    np.testing.assert_array_equal(num_samples,
        jnp.array([2, 2, 1], dtype=dtype))


def test_get_num_samples_zeros():
    num_classes = 3
    targets = jnp.array([0, 2, 2, 0, 0], dtype=jnp.int32)

    num_samples = get_num_samples(targets, num_classes, dtype=jnp.int32)

    assert num_samples.shape == (num_classes,)
    assert num_samples.dtype == jnp.int32

    np.testing.assert_array_equal(num_samples,
        jnp.array([3, 0, 2], dtype=jnp.int32))


def test_get_num_samples_vmap():
    num_classes = 3
    targets = jnp.array([
        [1, 0, 2, 1, 0],
        [0, 2, 2, 0, 0]
    ], dtype=jnp.int32)

    num_samples = vmap(get_num_samples, in_axes=(0, None, None))(
        targets, num_classes, jnp.int32)

    assert num_samples.shape == (2, num_classes)
    assert num_samples.dtype == jnp.int32

    np.testing.assert_array_equal(num_samples,
        jnp.array([[2, 2, 1], [3, 0, 2]], dtype=jnp.int32))


def test_get_prototypes():
    key = random.PRNGKey(0)
    num_classes = 3
    embeddings = random.normal(key, (5, 7), dtype=jnp.float32)
    targets = jnp.array([1, 0, 2, 1, 0], dtype=jnp.int32)

    prototypes = get_prototypes(embeddings, targets, num_classes)

    assert prototypes.shape == (num_classes, 7)
    assert prototypes.dtype == jnp.float32

    expected_prototypes = np.zeros((num_classes, 7), dtype=np.float32)
    expected_num_samples = np.zeros((num_classes,), dtype=np.float32)
    for i in range(5):
        k = targets[i]
        for j in range(7):
            expected_prototypes[k, j] += embeddings[i, j]
        expected_num_samples[k] += 1

    for i in range(num_classes):
        for j in range(7):
            expected_prototypes[i, j] /= expected_num_samples[i]

    np.testing.assert_allclose(prototypes, expected_prototypes)


def test_get_prototypes_zeros():
    key = random.PRNGKey(0)
    num_classes = 3
    embeddings = random.normal(key, (5, 7), dtype=jnp.float32)
    targets = jnp.array([0, 2, 2, 0, 0], dtype=jnp.int32)

    prototypes = get_prototypes(embeddings, targets, num_classes)

    assert prototypes.shape == (num_classes, 7)
    assert prototypes.dtype == jnp.float32
    np.testing.assert_array_equal(prototypes[1], 0.)


def test_get_prototypes_vmap():
    key = random.PRNGKey(0)
    num_classes = 3
    embeddings = random.normal(key, (2, 5, 7), dtype=jnp.float32)
    targets = jnp.array([
        [1, 0, 2, 1, 0],
        [0, 2, 2, 0, 0]
    ], dtype=jnp.int32)

    prototypes = vmap(get_prototypes, in_axes=(0, 0, None))(
        embeddings, targets, num_classes)

    assert prototypes.shape == (2, num_classes, 7)
    assert prototypes.dtype == jnp.float32
