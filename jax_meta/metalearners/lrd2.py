import jax.numpy as jnp

from jax import vmap, lax, nn

from jax_meta.utils.losses import binary_cross_entropy, cross_entropy
from jax_meta.utils.metrics import binary_accuracy, accuracy
from jax_meta.metalearners.base import MetaLearner


class LRD2Binary(MetaLearner):
    def __init__(self, encoder, lambda_=0.1, num_iterations=10):
        super().__init__()
        self.encoder = encoder
        self.lambda_ = lambda_
        self.num_iterations = num_iterations

    def loss(self, params, state, linear_params, inputs, targets, args):
        features, state = self.model.apply(params, state, inputs, *args)
        logits = jnp.matmul(features, linear_params)
        loss = jnp.mean(binary_cross_entropy(logits, targets))
        logs = {
            'loss': loss,
            'accuracy': binary_accuracy(logits, targets),
        }
        return loss, (state, logs)

    def adapt(self, params, state, inputs, targets, args):
        features, _ = self.model.apply(params, state, inputs, *args)
        def _irls_step(linear_params, _):
            num_samples = inputs.shape[0]
            I = jnp.eye(num_samples)

            # Predictions
            logits = jnp.matmul(features, linear_params)
            preds = nn.sigmoid(logits)
            s = preds * (1 - preds)

            A = jnp.matmul(features, features.T * s) + self.lambda_ * I
            b = logits + (targets - preds) / s
            solution = jnp.linalg.solve(A, b)

            logs = {
                'loss': jnp.mean(binary_cross_entropy(logits, targets)),
                'accuracy': binary_accuracy(logits, targets),
            }
            return jnp.dot(features.T * s, solution), logs

        init_params = jnp.zeros((features.shape[-1],), dtype=features.dtype)
        return lax.scan(_irls_step, init_params, None, length=self.num_iterations)

    def meta_init(self, key, *args, **kwargs):
        return self.model.init(key, *args, **kwargs)


class LRD2OneVsRest(LRD2Binary):
    def __init__(
            self,
            encoder,
            lambda_=0.1,
            num_iterations=10,
            num_ways=5
        ):
        super().__init__(
            encoder,
            lambda_=lambda_,
            num_iterations=num_iterations
        )
        self.num_ways = num_ways

    def loss(self, params, state, linear_params, inputs, targets, args):
        features, state = self.model.apply(params, state, inputs, *args)
        logits = jnp.matmul(features, linear_params)
        loss = jnp.mean(cross_entropy(logits, targets))
        logs = {
            'loss': loss,
            'accuracy': accuracy(logits, targets),
        }
        return loss, (state, logs)

    def adapt(self, params, state, inputs, targets, args):
        features, _ = self.model.apply(params, state, inputs, *args)
        one_hot_targets = nn.one_hot(targets, self.num_ways)
        def _irls_step(linear_params, _):
            num_samples = inputs.shape[0]
            I = jnp.eye(num_samples)

            # Predictions
            logits = jnp.dot(features, linear_params)
            preds = nn.sigmoid(logits)
            s = preds * (1 - preds)

            A = jnp.matmul(features, features.T) * jnp.expand_dims(s.T, axis=1) + self.lambda_ * I
            b = logits + (one_hot_targets - preds) / s
            solution = jnp.linalg.solve(A, b.T)

            logs = {
                'loss': jnp.mean(cross_entropy(logits, targets)),
                'accuracy': accuracy(logits, targets),
            }

            new_params = vmap(jnp.dot)(
                features.T * jnp.expand_dims(s.T, axis=1), solution)
            return new_params.T, logs

        init_params = jnp.zeros((features.shape[-1], self.num_ways), dtype=features.dtype)
        return lax.scan(_irls_step, init_params, None, length=self.num_iterations)
