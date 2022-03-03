import jax.numpy as jnp
import optax
import jax

from jax import vmap, grad, tree_util, jit, lax
from functools import partial

from jax_meta.metalearners.base import MetaLearnerState
from jax_meta.metalearners.maml import MAML


class iMAML(MAML):
    def __init__(
            self,
            model,
            num_steps=5,
            alpha=0.1,
            lambda_=1.,
            regu_coef=1.,
            cg_damping=10.,
            cg_steps=5
        ):
        super().__init__(model, num_steps=num_steps, alpha=alpha)
        self.lambda_ = lambda_
        self.regu_coef = regu_coef
        self.cg_damping = cg_damping
        self.cg_steps = cg_steps

    def adapt(self, init_params, state, inputs, targets, args):
        loss_grad = grad(self.loss, has_aux=True)
        # Gradient descent with proximal regularization
        gradient_descent = lambda p, p0, g: p - self.alpha * (g + self.lambda_ * (p - p0))
        def _gradient_update(params, _):
            # Do not update the state during adaptation
            grads, (_, logs) = loss_grad(params, state, inputs, targets, args)
            params = tree_util.tree_map(gradient_descent, params, init_params, grads)
            return params, logs

        return lax.scan(
            _gradient_update,
            init_params,
            None,
            length=self.num_steps
        )

    def hessian_vector_product(self, params, state, inputs, targets, args):
        train_loss = lambda primals: self.loss(primals, state, inputs, targets, args)[0]
        _, hvp_fn = jax.linearize(grad(train_loss), params)

        def _hvp_damping(tangents):
            damping = lambda h, t: (1. + self.regu_coef) * t + h / (self.lambda_ + self.cg_damping)
            return tree_util.tree_multimap(damping, hvp_fn(tangents), tangents)

        return _hvp_damping

    def grad_outer_loss(self, params, state, train, test, args):
        @partial(vmap, in_axes=(None, None, 0, 0))
        def _grad_outer_loss(params, state, train, test):
            adapted_params, inner_logs = self.adapt(
                params, state, train.inputs, train.targets, args
            )

            # Compute the gradient wrt. the adapted parameters
            grads, (state, outer_logs) = grad(self.loss, has_aux=True)(
                adapted_params, state, test.inputs, test.targets, args)

            # Compute the meta-gradient using Conjugate Gradient
            hvp_fn = self.hessian_vector_product(
                adapted_params, state, train.inputs, train.targets, args
            )
            outer_grads, _ = jax.scipy.sparse.linalg.cg(hvp_fn, grads, maxiter=self.cg_steps)

            return outer_grads, inner_logs, outer_logs, state

        outer_grads, inner_logs, outer_logs, states = _grad_outer_loss(
            params, state, train, test)
        outer_grads = tree_util.tree_map(partial(jnp.mean, axis=0), outer_grads)
        state = tree_util.tree_map(partial(jnp.mean, axis=0), states)
        
        logs = {
            **{f'inner/{k}': v for (k, v) in inner_logs.items()},
            **{f'outer/{k}': v for (k, v) in outer_logs.items()}
        }
        return outer_grads, (state, logs)

    @partial(jit, static_argnums=(0, 5))
    def train_step(self, params, state, train, test, args):
        grads, (model_state, logs) = self.grad_outer_loss(
            params, state.model, train, test, args)

        updates, opt_state = self.optimizer.update(grads, state.optimizer, params)
        params = optax.apply_updates(params, updates)

        state = MetaLearnerState(model=model_state, optimizer=opt_state)

        return params, state, logs
