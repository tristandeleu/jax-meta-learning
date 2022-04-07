import jax.numpy as jnp


def get_num_samples(targets, num_classes, dtype=None):
    num_samples = jnp.zeros((num_classes,), dtype=dtype)
    return num_samples.at[targets].add(1)


def get_prototypes(inputs, targets, num_classes):
    prototypes = jnp.zeros((num_classes, inputs.shape[-1]), dtype=inputs.dtype)
    prototypes = prototypes.at[targets].add(inputs)

    num_samples = get_num_samples(targets, num_classes, dtype=inputs.dtype)
    num_samples = jnp.expand_dims(jnp.maximum(num_samples, 1), axis=-1)

    return prototypes / num_samples
