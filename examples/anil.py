import jax.numpy as jnp
import haiku as hk
import optax
import os

from jax import random
from tqdm import tqdm

from jax_meta.modules.conv import Conv4
from jax_meta.metalearners.anil import ANIL
from jax_meta.utils.arguments import DataArguments


def main(args):
    @hk.without_apply_rng
    @hk.transform_with_state
    def encoder(inputs, is_training):
        return Conv4()(inputs, is_training)

    @hk.without_apply_rng
    @hk.transform_with_state
    def classifier(inputs, _):
        return hk.Linear(args.data.ways)(inputs)

    metalearner = ANIL(
        encoder,
        classifier,
        **args.metalearner_kwargs,
    )

    dataset = args.data.dataset._cls(
        args.data.folder,
        args.batch_size,
        shots=args.data.shots,
        ways=args.data.ways,
        test_shots=args.data.test_shots,
        size=args.num_batches,
        split='train',
        seed=args.data.seed,
        download=True
    )

    key = random.PRNGKey(args.data.seed)
    optimizer = optax.adam(args.meta_lr)
    params, state = metalearner.init(key, optimizer, dataset.dummy_input, True)

    dataset.reset()
    with tqdm(dataset, desc='Train') as pbar:
        for idx, batch in enumerate(pbar):
            params, state, results = metalearner.step(
                params, state, batch['train'], batch['test'], True)

            if (idx % 100) == 0:
                pbar.set_postfix(
                    loss=f'{results["after/loss"].mean():.4f}',
                    accuracy=f'{100 * results["after/accuracy"].mean():.2f}'
                )


if __name__ == '__main__':
    import json
    from simple_parsing import ArgumentParser

    parser = ArgumentParser('Almost No Inner-Loop (ANIL) algorithm.')
    parser.add_arguments(DataArguments, dest='data')
    
    parser.add_argument('--metalearner_kwargs', type=json.loads, default='{}',
        help='Keyword arguments to send to the meta-learner.')

    # Optimization
    optim = parser.add_argument_group('Optimization')
    optim.add_argument('--batch_size', type=int, default=4,
        help='number of tasks in a batch of tasks (default: %(default)s).')
    optim.add_argument('--num_batches', type=int, default=100,
        help='number of batch of tasks (default: %(default)s).')
    optim.add_argument('--meta_lr', type=float, default=1e-3,
        help='learning rate for the meta-optimizer (optimization of the outer '
        'loss). The default optimizer is Adam (default: %(default)s).')

    args = parser.parse_args()
    main(args)
