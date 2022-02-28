import jax.numpy as jnp
import haiku as hk
import math

from jax import nn, random
from collections import namedtuple

from jax_meta.utils.losses import cross_entropy
from jax_meta.utils.metrics import accuracy
from jax_meta.metalearners.base import MetaLearner
from jax_meta.modules.r2d2 import Adjust


R2D2MetaParameters = namedtuple('R2D2MetaParameters', ['encoder', 'adjust'])


class R2D2(MetaLearner):
    def __init__(self, encoder, adjust=None, num_ways=5, lambda_=0.1):
        super().__init__()

        if adjust is None:
            @hk.without_apply_rng
            @hk.transform
            def adjust(inputs):
                return Adjust(1)(inputs)

        self.encoder = encoder
        self.adjust = adjust
        self.num_ways = num_ways
        self.lambda_ = lambda_

    def loss(self, params, state, linear_params, inputs, targets, args):
        features, state = self.model.apply(params.encoder, state, inputs, *args)
        features = R2D2.append_ones(features)

        outputs = jnp.matmul(features, linear_params)
        logits = self.adjust.apply(params.adjust, outputs)
        loss = jnp.mean(cross_entropy(logits, targets))
        logs = {
            'loss': loss,
            'accuracy': accuracy(logits, targets),
        }
        return loss, (state, logs)

    def adapt(self, params, state, inputs, targets, args):
        num_samples = inputs.shape[0]
        I = jnp.eye(num_samples)

        features, _ = self.model.apply(params.encoder, state, inputs, *args)
        features = features / math.sqrt(num_samples)
        features = R2D2.append_ones(features)

        A = jnp.matmul(features, features.T) + self.lambda_ * I
        b = nn.one_hot(targets, self.num_ways) / math.sqrt(num_samples)
        solution = jnp.linalg.solve(A, b)

        return (jnp.matmul(features.T, solution), {})

    def meta_init(self, key, *args, **kwargs):
        subkey1, subkey2 = random.split(key)
        encoder, encoder_state = self.model.init(subkey1, *args, **kwargs)

        features = self.model.apply(encoder, encoder_state, *args, **kwargs)
        adjust = self.adjust.init(subkey2, features)

        params = R2D2MetaParameters(encoder=encoder, adjust=adjust)
        return params, encoder_state

    @staticmethod
    def append_ones(features):
        ones = jnp.ones(features.shape[:-1] + (1,), dtype=features.dtype)
        return jnp.append(features, ones, axis=-1)
