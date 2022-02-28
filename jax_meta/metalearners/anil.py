from jax import random
from collections import namedtuple

from jax_meta.metalearners.maml import MAML


ANILMetaParameters = namedtuple('ANILMetaParameters', ['encoder', 'classifier'])


class ANIL(MAML):
    def __init__(self, encoder, classifier, num_steps=5, alpha=0.01):
        super().__init__(classifier, num_steps=num_steps, alpha=alpha)
        self.encoder = encoder

    def outer_loss(self, params, state, train, test, args):
        train_features, _ = self.encoder.apply(
            params.encoder, state.encoder, train.inputs, *args
        )
        adapted_params, inner_logs = self.adapt(
            params.classifier,
            state.classifier,
            train_features,
            train.targets,
            args
        )

        test_features, state_encoder = self.encoder.apply(
            params.encoder, state.encoder, test.inputs, *args
        )
        outer_loss, (state_classifier, outer_logs) = self.loss(
            adapted_params,
            state.classifier,
            test_features,
            test.targets,
            args
        )

        state = ANILMetaParameters(
            encoder=state_encoder,
            classifier=state_classifier
        )
        return outer_loss, state, inner_logs, outer_logs

    def meta_init(self, key, inputs, *args, **kwargs):
        subkey1, subkey2 = random.split(key)

        params_encoder, state_encoder = self.encoder.init(subkey1, inputs, *args, **kwargs)
        features, _ = self.encoder.apply(params_encoder, state_encoder, inputs, *args, **kwargs)
        params_classifier, state_classifier = self.model.init(subkey2, features, *args, **kwargs)

        params = ANILMetaParameters(encoder=params_encoder, classifier=params_classifier)
        state = ANILMetaParameters(encoder=state_encoder, classifier=state_classifier)
        return (params, state)
