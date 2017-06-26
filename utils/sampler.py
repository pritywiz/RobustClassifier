import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

from . import util
import pickle
import os

class DataSampler(object):
    def __init__(self):
        self.mnist     = None
        self.channel   = 1
        self.image_dim = 28

    def __call__(self, split_type, batch_size = None):
        if self.mnist is None:
            self.mnist = input_data.read_data_sets('MNIST_data', reshape=False, one_hot = True)
        if split_type == "train":
            if batch_size is not None:
                samples = self.mnist.train.next_batch(batch_size)
                return util.pad_images(samples[0], self.channel, self.image_dim), samples[1]
            else:
                samples = self.mnist.train
                return util.pad_images(samples.images, self.channel, self.image_dim), samples.labels
        if split_type == "validation":
            samples = self.mnist.validation
            return util.pad_images(samples.images, self.channel, self.image_dim), samples.labels

        if split_type == "test":
            samples = self.mnist.test
            return util.pad_images(samples.images, self.channel, self.image_dim), samples.labels

    def perturbed(self, num_images = 5, batch_size = None):
        if self.mnist is None:
            self.mnist = input_data.read_data_sets('MNIST_data', reshape=False, one_hot = True)

        if batch_size is not None:
            samples = self.mnist.test.next_batch(batch_size)
            images = samples[0]
            w_rand = np.random.randint(images.shape[1], size=(num_images))
            h_rand = np.random.randint(images.shape[2], size=(num_images))
            images[:, w_rand, h_rand, :] = np.random.rand(1)

            return util.pad_images(images, self.channel, self.image_dim), samples[1]

        else:
            samples = self.mnist.test
            images = samples.images
            w_rand = np.random.randint(samples.images.shape[1], size=(num_images))
            h_rand = np.random.randint(samples.images.shape[2], size=(num_images))
            images[:, w_rand, h_rand, :] = np.random.rand(1)

            return util.pad_images(images, self.channel, self.image_dim), samples.labels

    def generated(self, r_dir = "/input", filename = "gen_img.p"):
        gen_data = pickle.load( open( os.path.join(r_dir, filename), "rb" ) )

        gen_images = np.reshape(np.array(gen_data["images"]), (-1, 1, 32, 32))
        gen_clabels = np.array(gen_data["clabels"])
        gen_glabels = np.array(gen_data["glabels"])

        return gen_images, gen_clabels, gen_glabels


    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]

class NoiseSampler(object):
    def __call__(self, batch_size, z_dim):
        return np.random.uniform(-1.0, 1.0, [batch_size, z_dim]).astype(np.float32)

