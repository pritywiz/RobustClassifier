import tensorflow as tf
import numpy as np
import tensorflow.contrib.layers as ly

class Classifier(object):
    def __init__(self, model_dim, data_format, label_dim):
        self.model_dim   = model_dim
        self.data_format = data_format
        self.name        = 'classifier'
        self.label_dim   = label_dim

    def __call__(self, inputs, reuse=True):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()


            # first layer with Input Dimension (1 X 32 X 32) and output dimension (32 X 32 X 32)
            deepnet1 = ly.conv2d(inputs, num_outputs=self.model_dim, kernel_size=3, data_format=self.data_format, 
                      stride=1, activation_fn=tf.nn.relu, normalizer_fn=ly.batch_norm, 
                      normalizer_params={'fused': True, 'data_format': self.data_format})


            deepnet2 = ly.conv2d(deepnet1, num_outputs=self.model_dim, kernel_size=3, data_format=self.data_format, 
                      stride=1, activation_fn=tf.nn.relu, normalizer_fn=ly.batch_norm, 
                      normalizer_params={'fused': True, 'data_format': self.data_format})
            deepnet2 = ly.max_pool2d(deepnet2, 2, stride=2, padding='SAME', data_format=self.data_format)


            fl1 = ly.flatten(deepnet2)

            # fully connect
            fc1 = ly.fully_connected(fl1, 128, activation_fn=tf.nn.relu, normalizer_fn=ly.batch_norm)

            cat = ly.fully_connected(fc1, self.label_dim, activation_fn=None, normalizer_fn=ly.batch_norm)
            return cat

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]

