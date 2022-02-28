import jax.numpy as jnp

from jax import nn


def nll_loss(log_likelihoods, targets):
    losses = jnp.take_along_axis(log_likelihoods, targets[..., None], axis=-1)
    return -jnp.squeeze(losses, axis=-1)


def cross_entropy(logits, targets):
    log_likelihoods = nn.log_softmax(logits, axis=-1)
    return nll_loss(log_likelihoods, targets)


def binary_cross_entropy(logits, targets):
    # See: https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits
    return nn.relu(logits) - logits * targets + nn.softplus(-jnp.abs(logits))
