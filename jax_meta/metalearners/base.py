import jax.numpy as jnp
import optax
import jax

from abc import ABC, abstractmethod
from functools import partial
from collections import namedtuple


MetaLearnerState = namedtuple('MetaLearnerState', ['model', 'optimizer', 'key'])


class MetaLearner(ABC):
    def __init__(self):
        self._optimizer = None
        self._train_step = None

        self._batch_outer_loss = jax.jit(
            self.batch_outer_loss,
            static_argnums=(4,)
        )

    @abstractmethod
    def loss(self, params, state, inputs, targets, args):
        pass

    @abstractmethod
    def adapt(self, params, state, inputs, targets, args):
        pass

    @abstractmethod
    def meta_init(self, key, *args, **kwargs):
        pass

    def outer_loss(self, params, state, train, test, args):
        adapted_params, inner_logs = self.adapt(
            params, state, train.inputs, train.targets, args
        )
        outer_loss, (state, outer_logs) = self.loss(
            adapted_params, state, test.inputs, test.targets, args
        )
        return (outer_loss, state, inner_logs, outer_logs)

    def train_step(self, params, state, train, test, args):
        outer_loss_grad = jax.grad(self.batch_outer_loss, has_aux=True)
        grads, (model_state, logs) = outer_loss_grad(
            params, state.model, train, test, args)

        # Apply gradient descent
        updates, opt_state = self.optimizer.update(grads, state.optimizer, params)
        params = optax.apply_updates(params, updates)

        # TODO: The key will eventually be split and used in various methods
        state = MetaLearnerState(model=model_state, optimizer=opt_state, key=state.key)

        return params, state, logs

    @property
    def _step(self):
        if self._train_step is None:
            # Only compile the training step function once
            self._train_step = jax.jit(self.train_step, static_argnums=(4,))
        return self._train_step

    def step(self, params, state, train, test, *args):
        return self._step(params, state, train, test, args)

    @property
    def optimizer(self):
        if self._optimizer is None:
            raise ValueError(f'`{self:s}` contains no optimizer. To train the'
                             'model, you must call the `init` function.')
        return self._optimizer

    def batch_outer_loss(self, params, state, train, test, args):
        outer_loss = jax.vmap(self.outer_loss, in_axes=(None, None, 0, 0, None))
        outer_losses, states, inner_logs, outer_logs = outer_loss(
            params, state, train, test, args
        )
        state = jax.tree_util.tree_map(partial(jnp.mean, axis=0), states)

        logs = {
            **{f'inner/{k}': v for (k, v) in inner_logs.items()},
            **{f'outer/{k}': v for (k, v) in outer_logs.items()}
        }
        return jnp.mean(outer_losses), (state, logs)

    def init(self, key, optimizer, *args, **kwargs):
        self._optimizer = optimizer
        key, subkey = jax.random.split(key)
        params, model_state = self.meta_init(subkey, *args, **kwargs)
        state = MetaLearnerState(
            model=model_state,
            optimizer=self.optimizer.init(params),
            key=key
        )
        return params, state

    def evaluate(self, params, state, dataset, *args):
        if dataset.size is None:
            raise RuntimeError('The dataset for evaluation must be finite, '
                'got an infinite dataset. You must set the `size` argument '
                'when creating the dataset.')

        results = None
        for i, batch in enumerate(dataset.reset()):
            _, (_, logs) = self._batch_outer_loss(
                params,
                state.model,
                batch['train'],
                batch['test'],
                args
            )
            logs = jax.tree_util.tree_map(
                lambda arr: jnp.mean(arr, axis=0), logs)

            if results is None:
                results = logs
            else:
                online_mean = lambda mean, arr: mean + ((arr - mean) / (i + 1))
                results = jax.tree_util.tree_map(online_mean, results, logs)

        return results
