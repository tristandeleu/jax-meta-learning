import haiku as hk
import math

from jax import nn


class ConvBlock(hk.Module):
    def __init__(self, channels, name=None):
        super().__init__(name=name)
        self.channels = channels

    def __call__(self, inputs, is_training):
        outputs = inputs
        outputs = hk.Conv2D(self.channels, kernel_shape=3,
            stride=1, with_bias=True, name='conv')(outputs)
        outputs = hk.BatchNorm(create_scale=True, create_offset=True,
            decay_rate=0.9, name='norm')(outputs, is_training)
        outputs = nn.relu(outputs)
        outputs = hk.max_pool(outputs, 2, 2, padding='VALID')
        return outputs


class Conv4(hk.Module):
    def __init__(self, num_filters=64, normalize_outputs=False, name=None):
        super().__init__(name=name)
        self.num_filters = num_filters
        self.normalize_outputs = normalize_outputs

    def __call__(self, inputs, is_training):
        outputs = inputs
        outputs = ConvBlock(self.num_filters, name='layer1')(outputs, is_training)
        outputs = ConvBlock(self.num_filters, name='layer2')(outputs, is_training)
        outputs = ConvBlock(self.num_filters, name='layer3')(outputs, is_training)
        outputs = ConvBlock(self.num_filters, name='layer4')(outputs, is_training)
        outputs = outputs.reshape(inputs.shape[:-3] + (-1,))
        normalization = math.sqrt(outputs.shape[-1]) if self.normalize_outputs else 1.
        return outputs / normalization
