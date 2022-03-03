import jax.numpy as jnp
import optax

from abc import ABC, abstractmethod
from jax import jit, grad, tree_util, vmap
from functools import partial
from collections import namedtuple


MetaLearnerState = namedtuple('MetaLearnerState', ['model', 'optimizer'])


class MetaLearner(ABC):
    def __init__(self):
        self._optimizer = None

    @abstractmethod
    def loss(self, params, state, inputs, targets, args):
        pass

    @abstractmethod
    def adapt(self, params, state, inputs, targets, args):
        pass

    @abstractmethod
    def meta_init(self, key, *args, **kwargs):
        pass

    def step(self, params, state, train, test, *args):
        return self.train_step(params, state, train, test, args)

    def outer_loss(self, params, state, train, test, args):
        adapted_params, inner_logs = self.adapt(
            params, state, train.inputs, train.targets, args
        )
        outer_loss, (state, outer_logs) = self.loss(
            adapted_params, state, test.inputs, test.targets, args
        )
        return (outer_loss, state, inner_logs, outer_logs)

    @partial(jit, static_argnums=(0, 5))
    def train_step(self, params, state, train, test, args):
        outer_loss_grad = grad(self.batch_outer_loss, has_aux=True)
        grads, (model_state, logs) = outer_loss_grad(
            params, state.model, train, test, args)

        # Apply gradient descent
        updates, opt_state = self.optimizer.update(grads, state.optimizer, params)
        params = optax.apply_updates(params, updates)

        state = MetaLearnerState(model=model_state, optimizer=opt_state)

        return params, state, logs

    @property
    def optimizer(self):
        if self._optimizer is None:
            raise ValueError(f'`{self:s}` contains no optimizer. To train the'
                             'model, you must call the `init` function.')
        return self._optimizer

    def batch_outer_loss(self, params, state, train, test, args):
        outer_loss = vmap(self.outer_loss, in_axes=(None, None, 0, 0, None))
        outer_losses, states, inner_logs, outer_logs = outer_loss(
            params, state, train, test, args
        )
        state = tree_util.tree_map(partial(jnp.mean, axis=0), states)

        logs = {
            **{f'inner/{k}': v for (k, v) in inner_logs.items()},
            **{f'outer/{k}': v for (k, v) in outer_logs.items()}
        }
        return jnp.mean(outer_losses), (state, logs)

    def init(self, key, optimizer, *args, **kwargs):
        self._optimizer = optimizer
        params, model_state = self.meta_init(key, *args, **kwargs)
        state = MetaLearnerState(
            model=model_state,
            optimizer=self.optimizer.init(params)
        )
        return params, state

    def evaluate(self, params, state, dataset, *args):
        if dataset.size is None:
            raise RuntimeError('The dataset for evaluation must be finite, '
                'got an infinite dataset. You must set the `size` argument '
                'when creating the dataset.')

        outer_loss = jit(self.batch_outer_loss, static_argnums=(4,))
        results = None
        for i, batch in enumerate(dataset.reset()):
            _, (_, logs) = outer_loss(params, state.model, batch['train'], batch['test'], args)
            logs = tree_util.tree_map(lambda arr: jnp.mean(arr, axis=0), logs)

            if results is None:
                results = logs
            else:
                online_mean = lambda mean, arr: mean + ((arr - mean) / (i + 1))
                results = tree_util.tree_map(online_mean, results, logs)

        return results
