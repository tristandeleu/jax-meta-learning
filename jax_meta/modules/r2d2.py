import haiku as hk
import jax.numpy as jnp


class Adjust(hk.Module):
    def __init__(self, base, scale_init=1e-4, bias_init=0., name=None):
        super().__init__(name=name)
        self.base = base
        self._scale_init = scale_init
        self._bias_init = bias_init

    def __call__(self, inputs):
        # Parameters
        scale_init = hk.initializers.Constant(self._scale_init)
        scale = hk.get_parameter('scale', (), inputs.dtype, init=scale_init)

        bias_init = hk.initializers.Constant(self._bias_init)
        bias = hk.get_parameter('bias', (), inputs.dtype, init=bias_init)

        if self.base == 1:
            return inputs * scale + bias
        else:
            return inputs * (self.base ** scale) + self.base ** bias - 1
