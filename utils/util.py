import numpy as np
import tensorflow as tf

def pad_images(images, channel, image_dim, pad = 2):
    images = 2 * images - 1

    images = np.reshape(images, (-1, channel, image_dim, image_dim))

    npad = ((0, 0), (0, 0), (pad, pad), (pad, pad))
    images = np.pad(images, pad_width=npad, mode='constant', constant_values=-1)
    return images


def LeakyReLU(x, leak=0.2, name="lrelu"): # Relu has defined in https://github.com/tensorflow/tensorflow/issues/4079#issuecomment-243318490
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

