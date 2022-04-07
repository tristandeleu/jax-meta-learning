import pytest

import numpy as np
import jax.numpy as jnp

from jax import random, vmap

from jax_meta.utils.matching import (pairwise_cosine_similarity,
    matching_log_probas, matching_probas)


def test_pairwise_cosine_similarity():
    key = random.PRNGKey(0)
    eps = 1e-8

    key, subkey = random.split(key)
    embeddings1 = random.normal(subkey, (3, 7))

    key, subkey = random.split(key)
    embeddings2 = random.normal(subkey, (5, 7))

    similarities = pairwise_cosine_similarity(embeddings1, embeddings2, eps=eps)

    assert similarities.shape == (3, 5)

    norm1 = np.sqrt(np.sum(embeddings1 ** 2, axis=1))
    norm2 = np.sqrt(np.sum(embeddings2 ** 2, axis=1))
    expected_similarities = np.zeros((3, 5), dtype=np.float32)
    for i in range(3):
        for j in range(5):
            for k in range(7):
                expected_similarities[i, j] += embeddings1[i, k] * embeddings2[j, k]
            expected_similarities[i, j] /= max(norm1[i] * norm2[j], eps)

    np.testing.assert_allclose(similarities, expected_similarities, atol=1e-7)


def test_pairwise_cosine_similarity_zeros():
    key = random.PRNGKey(0)
    eps = 1e-8

    key, subkey = random.split(key)
    embeddings1 = random.normal(subkey, (3, 7))
    embeddings1 = embeddings1.at[1].set(0.)  # Zero out one embedding

    key, subkey = random.split(key)
    embeddings2 = random.normal(subkey, (5, 7))

    similarities = pairwise_cosine_similarity(embeddings1, embeddings2, eps=eps)

    assert similarities.shape == (3, 5)
    np.testing.assert_array_equal(similarities[1], 0.)


def test_matching_log_probas():
    key = random.PRNGKey(0)
    eps = 1e-8
    num_classes = 3

    key, subkey = random.split(key)
    embeddings = random.normal(subkey, (5, 11))

    key, subkey = random.split(key)
    test_embeddings = random.normal(subkey, (7, 11))

    targets = jnp.array([1, 0, 2, 2, 1], dtype=jnp.int32)

    log_probas = matching_log_probas(
        embeddings,
        targets,
        test_embeddings,
        num_classes,
        eps=eps
    )

    assert log_probas.shape == (num_classes, 7)
    np.testing.assert_array_less(log_probas, 0.)


def test_matching_log_probas_vmap():
    key = random.PRNGKey(0)
    eps = 1e-8
    num_classes = 3

    key, subkey = random.split(key)
    embeddings = random.normal(subkey, (2, 5, 11))

    key, subkey = random.split(key)
    test_embeddings = random.normal(subkey, (2, 7, 11))

    targets = jnp.array([
        [1, 0, 2, 2, 1],
        [0, 2, 2, 0, 0]
    ], dtype=jnp.int32)

    log_probas = vmap(matching_log_probas, (0, 0, 0, None, None))(
        embeddings, targets, test_embeddings, num_classes, eps)

    assert log_probas.shape == (2, num_classes, 7)
    np.testing.assert_array_less(log_probas, 0.)


def test_matching_probas():
    key = random.PRNGKey(0)
    eps = 1e-8
    num_classes = 3

    key, subkey = random.split(key)
    embeddings = random.normal(subkey, (5, 11))

    key, subkey = random.split(key)
    test_embeddings = random.normal(subkey, (7, 11))

    targets = jnp.array([1, 0, 2, 2, 1], dtype=jnp.int32)

    probas = matching_probas(
        embeddings,
        targets,
        test_embeddings,
        num_classes,
        eps=eps
    )

    assert probas.shape == (num_classes, 7)
    assert jnp.all(0. <= probas)
    assert jnp.all(probas <= 1)
    np.testing.assert_allclose(probas.sum(0), 1., atol=1e-7)

    similarities = pairwise_cosine_similarity(embeddings, test_embeddings, eps=eps)
    exp_similarities = jnp.exp(similarities)

    expected_probas = np.zeros((num_classes, 7), dtype=np.float32)
    for i in range(5):
        k = targets[i]
        for j in range(7):
            expected_probas[k, j] += (exp_similarities[i, j]
                / np.sum(exp_similarities[:, j]))

    np.testing.assert_allclose(probas, expected_probas, atol=1e-7)


def test_matching_probas_vmap():
    key = random.PRNGKey(0)
    eps = 1e-8
    num_classes = 3

    key, subkey = random.split(key)
    embeddings = random.normal(subkey, (2, 5, 11))

    key, subkey = random.split(key)
    test_embeddings = random.normal(subkey, (2, 7, 11))

    targets = jnp.array([
        [1, 0, 2, 2, 1],
        [0, 2, 2, 0, 0]
    ], dtype=jnp.int32)

    probas = vmap(matching_probas, (0, 0, 0, None, None))(
        embeddings, targets, test_embeddings, num_classes, eps)

    assert probas.shape == (2, num_classes, 7)
    assert jnp.all(0. <= probas)
    assert jnp.all(probas <= 1.)
    np.testing.assert_allclose(probas.sum(1), 1., atol=1e-7)