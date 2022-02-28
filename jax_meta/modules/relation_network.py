import jax.numpy as jnp
import haiku as hk


class RelationNetwork(hk.nets.MLP):
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        num_inputs = inputs.shape[-2]

        left = jnp.expand_dims(inputs, axis=-2).repeat(num_inputs, axis=-2)
        right = jnp.expand_dims(inputs, axis=-3).repeat(num_inputs, axis=-3)
        concatenated = jnp.concatenate([left, right], axis=-1)

        outputs =  super().__call__(concatenated)
        return jnp.mean(outputs, axis=-2)
