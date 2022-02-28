import pytest
import haiku as hk
import optax
import jax.numpy as jnp

from jax import random

from jax_meta.metalearners.base import MetaLearner, MetaLearnerState
from jax_meta.metalearners.maml import MAML


@pytest.fixture
def model():
    def network(inputs, _):
        return hk.Linear(5)(inputs)
    return hk.without_apply_rng(hk.transform_with_state(network))


def test_maml(model):
    metalearner = MAML(model, num_steps=1, alpha=0.1)
    assert isinstance(metalearner, MetaLearner)


def test_maml_init(model):
    metalearner = MAML(model, num_steps=1, alpha=0.1)
    optimizer = optax.adam(1e-3)

    key = random.PRNGKey(0)
    inputs = jnp.zeros((2, 3))
    _, state = metalearner.init(key, optimizer, inputs, True)

    assert isinstance(state, MetaLearnerState)
