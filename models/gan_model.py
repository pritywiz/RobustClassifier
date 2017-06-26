import tensorflow as tf
import tensorflow.contrib.layers as ly
from utils import util

class Discriminator(object):
    def __init__(self, model_dim, data_format):
        self.model_dim = model_dim
        self.data_format = data_format
        self.image_dim = 32
        self.name = 'discriminator'

    def __call__(self, inputs, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            dconv1 = ly.conv2d(inputs, num_outputs=self.model_dim, kernel_size=3, data_format=self.data_format,
                        stride=2, activation_fn=util.LeakyReLU)

            dconv2 = ly.conv2d(dconv1, num_outputs=self.model_dim * 2, kernel_size=3, data_format=self.data_format,
                        stride=2, activation_fn=util.LeakyReLU)

            dconv3 = ly.conv2d(dconv2, num_outputs=self.model_dim * 4, kernel_size=3, data_format=self.data_format,
                        stride=2, activation_fn=util.LeakyReLU, normalizer_fn=None)

            dconv4 = ly.conv2d(dconv3, num_outputs=self.model_dim * 8, kernel_size=3, data_format=self.data_format,
                        stride=2, activation_fn=util.LeakyReLU, normalizer_fn=None)

            dconv4 = ly.flatten(dconv4)

            disc = ly.fully_connected(dconv4, 1, activation_fn=None)

            return disc

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]


class Generator(object):
    def __init__(self, noise_dim, model_dim, channel, data_format):
        self.model_dim = model_dim
        self.noise_dim = noise_dim
        self.channel = channel
        self.data_format = data_format
        self.name = 'generator'

    def __call__(self, noise):
        with tf.variable_scope(self.name) as scope:
 
            fc1 = ly.fully_connected(noise, 4*4*4*2*self.model_dim, activation_fn=tf.nn.relu, normalizer_fn=None)
            fc1 = tf.reshape(fc1, [-1, 4*2*self.model_dim, 4, 4])

            convt1 = ly.conv2d_transpose(fc1, 4*self.model_dim, 3, stride=2, data_format=self.data_format,
                                activation_fn=tf.nn.relu, normalizer_fn=None, padding='SAME')

            convt2 = ly.conv2d_transpose(convt1, 2*self.model_dim, 3, stride=2, data_format=self.data_format,
                                activation_fn=tf.nn.relu, normalizer_fn=None, padding='SAME')

            convt3 = ly.conv2d_transpose(convt2, self.model_dim, 3, stride=2, data_format=self.data_format,
                                activation_fn=tf.nn.relu, normalizer_fn=None, padding='SAME')

            output = ly.conv2d_transpose(convt3, self.channel, 3, stride=1, data_format=self.data_format,
                                activation_fn=tf.nn.tanh, normalizer_fn=None, padding='SAME')

            return output

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]
