import numpy as np

from numpy.random import default_rng


def random_crop(data, size, padding, rng=default_rng()):
    x = rng.integers(2 * padding, size=data.shape[:-3] + (1, 1, 1))
    y = rng.integers(2 * padding, size=data.shape[:-3] + (1, 1, 1))
    arange = np.arange(size)
    rows = x + arange.reshape((size, 1, 1))
    cols = y + arange.reshape((1, size, 1))

    padding = (padding, padding)
    data = np.pad(data, ((0, 0),) * (data.ndim - 3) + (padding, padding, (0, 0)))
    data = np.take_along_axis(data, rows, axis=-3)
    data = np.take_along_axis(data, cols, axis=-2)
    return data


def random_horizontal_flip(data, rng=default_rng()):
    to_flip = (rng.random(size=data.shape[:-3]) < 0.5)
    data[to_flip] = np.flip(data[to_flip], axis=-2)
    return data


def normalize(data, mean, std):
    data -= mean
    data /= std
    return data


def _blend(img1, img2, ratio):
    ratio = ratio.reshape((-1,) + (1,) * (img1.ndim - 1))
    img1 *= ratio
    img1 += (1. - ratio) * img2
    img1 = np.clip(img1, 0., 1., out=img1)
    return img1


def rgb_to_grayscale(data):
    col = np.array([0.2989, 0.587, 0.114], dtype=data.dtype)
    return np.expand_dims(data.dot(col), axis=-1)


def adjust_brightness(data, factor):
    return _blend(data, np.zeros_like(data), factor)


def adjust_saturation(data, factor):
    gray = rgb_to_grayscale(data)
    return _blend(data, gray, factor)


def adjust_contrast(data, factor):
    mean_gray = np.mean(rgb_to_grayscale(data), axis=(-3, -2, -1), keepdims=True)
    return _blend(data, mean_gray, factor)


def color_jitter(data, brightness, contrast, saturation, rng=default_rng()):
    order = np.argsort(rng.random(size=(3,) + data.shape[:-3]), axis=0)
    brightness = rng.uniform(1. - brightness, 1. + brightness, size=data.shape[:-3])
    contrast = rng.uniform(1. - contrast, 1. + contrast, size=data.shape[:-3])
    saturation = rng.uniform(1. - saturation, 1. + saturation, size=data.shape[:-3])
    for transform in order:
        data[transform == 0] = adjust_brightness(data[transform == 0], brightness[transform == 0])
        data[transform == 1] = adjust_contrast(data[transform == 1], contrast[transform == 1])
        data[transform == 2] = adjust_saturation(data[transform == 2], saturation[transform == 2])
    return data
