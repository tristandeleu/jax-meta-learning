import jax.numpy as jnp


def accuracy(outputs, targets):
    return jnp.mean(jnp.argmax(outputs, axis=-1) == targets)


def binary_accuracy(logits, targets):
    return jnp.mean((logits > 0).astype(targets.dtype) == targets)
