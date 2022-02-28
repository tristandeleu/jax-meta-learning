import jax.numpy as jnp
import haiku as hk
import math

from jax import nn


class ResidualBlock(hk.Module):
    def __init__(self, channels, stride, name=None):
        super().__init__(name=name)
        self.channels = channels
        self.stride = stride

    def __call__(self, inputs, is_training):
        residuals = outputs = inputs

        outputs = hk.Conv2D(self.channels, kernel_shape=3,
            stride=1, with_bias=False, name='conv1')(outputs)
        outputs = hk.BatchNorm(create_scale=True, create_offset=True,
            decay_rate=0.9, name='norm1')(outputs, is_training)
        outputs = nn.leaky_relu(outputs, negative_slope=0.1)

        outputs = hk.Conv2D(self.channels, kernel_shape=3,
            stride=1, with_bias=False, name='conv2')(outputs)
        outputs = hk.BatchNorm(create_scale=True, create_offset=True,
            decay_rate=0.9, name='norm2')(outputs, is_training)
        outputs = nn.leaky_relu(outputs, negative_slope=0.1)

        outputs = hk.Conv2D(self.channels, kernel_shape=3,
            stride=1, with_bias=False, name='conv3')(outputs)
        outputs = hk.BatchNorm(create_scale=True, create_offset=True,
            decay_rate=0.9, name='norm3')(outputs, is_training)

        residuals = hk.Conv2D(self.channels, kernel_shape=1,
            stride=1, with_bias=False, name='downsample_conv')(residuals)
        residuals = hk.BatchNorm(create_scale=True, create_offset=True,
            decay_rate=0.9, name='downsample_norm')(residuals, is_training)

        outputs = outputs + residuals
        outputs = nn.leaky_relu(outputs, negative_slope=0.1)
        outputs = hk.max_pool(outputs, self.stride, self.stride, padding='VALID')

        return outputs


class ResNet12(hk.Module):
    def __init__(self, num_filters=64, normalize_outputs=False, name=None):
        super().__init__(name=name)
        self.num_filters = num_filters  # Unused
        self.normalize_outputs = normalize_outputs

    def __call__(self, inputs, is_training):
        outputs = inputs
        outputs = ResidualBlock(64, 2, name='block1')(outputs, is_training)
        outputs = ResidualBlock(160, 2, name='block2')(outputs, is_training)
        outputs = ResidualBlock(320, 2, name='block3')(outputs, is_training)
        outputs = ResidualBlock(640, 2, name='block4')(outputs, is_training)
        outputs = jnp.mean(outputs, axis=(1, 2))
        normalization = math.sqrt(outputs.shape[-1]) if self.normalize_outputs else 1.
        return outputs / normalization
