import jax.numpy as jnp

from jax import grad, tree_util, lax
from collections import namedtuple

from jax_meta.metalearners.maml import MAML


MetaSGDMetaParameters = namedtuple('MetaSGDMetaParameters', ['model', 'alpha'])


class MetaSGD(MAML):
    def adapt(self, params, state, inputs, targets, args):
        loss_grad = grad(self.loss, has_aux=True)
        gradient_descent = lambda p, a, g: p - a * g  # Gradient descent
        def _gradient_update(params, _):
            # Do not update the state during adaptation
            grads, (_, logs) = loss_grad(params.model, state, inputs, targets, args)
            params = tree_util.tree_multimap(
                gradient_descent,
                params.model,
                params.alpha,
                grads
            )
            return params, logs

        return lax.scan(_gradient_update, params, None, length=self.num_steps)

    def meta_init(self, key, *args, **kwargs):
        model, model_state = self.model.init(key, *args, **kwargs)
        alpha = tree_util.tree_map(lambda x: jnp.full_like(x, self.alpha), model)
        params = MetaSGDMetaParameters(model=model, alpha=alpha)
        return params, model_state
