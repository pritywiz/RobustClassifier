import os, sys
sys.path.append(os.getcwd())

import cv2
import numpy as np
import pickle
import random
import tensorflow as tf
import time


from models import gan_model as model
from models import mnist_model as mmodel

from utils import log
from utils import sampler as data

class WassersteinGAN(object):
    def __init__(self, generator_model, discriminator_model, classifier_model, input_data, noise_data, scale=10.0):
        self.gen_model  = generator_model
        self.disc_model = discriminator_model
        self.c_model    = classifier_model
        self.input_data = input_data
        self.noise_data = noise_data
        self.image_dim  = self.disc_model.image_dim
        self.label_dim  = self.c_model.label_dim
        self.noise_dim  = self.gen_model.noise_dim
        self.channel    = self.gen_model.channel

        self.real_data  = tf.placeholder(dtype=tf.float32, shape=(None, self.channel, self.image_dim, self.image_dim))
        self.real_label = tf.placeholder(dtype=tf.float32, shape=(None, self.label_dim))

        # fake data
        self.fake_label = tf.placeholder(dtype=tf.float32, shape=(None, self.label_dim))
        self.fake_input = tf.placeholder(dtype=tf.float32, shape=(None, self.noise_dim))
        self.fakez      = tf.concat([self.fake_label, self.fake_input], 1)


        self.fake_data  = self.gen_model(self.fakez)

        self.disc_real  = self.disc_model(self.real_data)
        self.disc_fake  = self.disc_model(self.fake_data, reuse=True)

        # Classifier
        self.cat_real        = self.c_model(self.real_data)
        self.cat_fake        = self.c_model(self.fake_data, reuse=True)

        # Loss - Wasserstein
        self.g_loss     = - tf.reduce_mean(self.disc_fake)
        self.d_loss     = tf.reduce_mean(self.disc_fake) - tf.reduce_mean(self.disc_real)

        alpha           = tf.random_uniform([], 0.0, 1.0)
        real_data_hat   = alpha * self.real_data + (1.0 - alpha) * self.fake_data 
        disc_hat        = self.disc_model(real_data_hat, reuse=True)
        gradients       = tf.gradients(disc_hat, real_data_hat)
        gradient_p      = scale * tf.square(tf.norm(gradients[0], ord=2) - 1.0)

        self.d_loss     = self.d_loss + gradient_p

        # Categorical Loss
        self.loss_c_f = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.cat_fake, labels=self.fake_label))
        loss_c_r = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.cat_real, labels=self.real_label))
        self.loss_c = (loss_c_r + self.loss_c_f) / 2

        self.optimizer_d= None
        self.optimizer_g= None
        self.optim_c_r  = None
        self.optim_c_f  = None

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.optimizer_d = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9)\
                .minimize(self.d_loss, var_list=self.disc_model.vars)
            self.optimizer_g = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9)\
                .minimize(self.g_loss, var_list=self.gen_model.vars)
            self.optim_c_f   = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9)\
                .minimize(self.loss_c_f, var_list=self.gen_model.vars)
            self.optim_c_r   = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9)\
                .minimize(self.loss_c, var_list=self.c_model.vars)

        self.sess = tf.Session()

    def train(self, batch_size=64, epochs=1000000, log_metrics=False, w_dir = "/output", save_chkp=True):

        self.sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        start_time = time.time()
        plot_data = {"d_loss": 0.0, "g_loss": 0.0, "c_loss_r": 0.0, "c_loss_f":0.0}

        if log_metrics:
            log_writer = log.LogWriter(w_dir, "wgan_train.csv")
            log_writer(plot_data)

        def next_feed_dict(iter):
            train_img, train_label = self.input_data("train", batch_size)
            batch_noise = self.noise_data(batch_size, self.noise_dim)

            # Generate random one-hot vectors as class condition
            if iter % 2 == 0:
                idx = np.random.random_integers(0, self.label_dim - 1, size=(batch_size,))
            else:
                idx = np.random.random_integers(0, self.label_dim - 1)
            label_f = np.zeros((batch_size, self.label_dim))
            label_f[np.arange(batch_size), idx] = 1


            feed_dict = {self.real_data: train_img, self.real_label: train_label, 
                           self.fake_input: batch_noise, self.fake_label: label_f }
            return feed_dict

        for i in range(0, epochs):
            d_iters = 5
            c_iterv = 2
            log_data = ""
 
            # Iterate Discriminator #
            for _ in range(0, d_iters):
                feed_dict = next_feed_dict(i)
                _, _d_loss = self.sess.run([self.optimizer_d, self.d_loss], feed_dict=feed_dict)
                plot_data["d_loss"] = _d_loss

            # Train  #
            batch_noise = self.noise_data(batch_size, self.noise_dim)
            train_img, train_label = self.input_data("train", batch_size)
            feed_dict = next_feed_dict(i)
            _, _g_loss = self.sess.run([self.optimizer_g, self.g_loss], feed_dict=feed_dict)
            plot_data["g_loss"] = _g_loss

            # Train Classier #
            if i % c_iterv == 0:
                feed_dict = next_feed_dict(i)

                _, _loss_c_f = self.sess.run([self.optim_c_f, self.loss_c_f], feed_dict=feed_dict)
                plot_data["c_loss_f"] = _loss_c_f

                _, _loss_c   = self.sess.run([self.optim_c_r, self.loss_c ], feed_dict=feed_dict)
                plot_data["c_loss_r"] = _loss_c

            if i % 100 == 0:
                log_data += 'Iter: {} '.format(i) 
                log_data += 'Time: {:>5.4f} '.format(time.time() - start_time)
                log_data += 'Disc Loss: {:>6.4f} '.format(plot_data["d_loss"]) 
                log_data += 'Gen Loss: {:>6.4f} '.format(plot_data["g_loss"]) 
                log_data += 'C Loss: {:>6.4f} '.format(plot_data["c_loss_r"]) 
                log_data += 'C Fake Loss: {:>6.4f} '.format(plot_data["c_loss_f"]) 

                print(log_data)

            if i % 1000 == 999 and save_chkp == True:
                save_path = saver.save(self.sess, os.path.join(w_dir, "wgan_model.ckpt"), global_step=i)
                print("Model saved in file: {}".format(save_path))
                self.generate(from_chk = False, batch_size=16, save_img = True, w_dir = w_dir, step = i)
                print("Image Generated")

            if log_metrics:
                log_writer.writeLog(plot_data)
                if i % 10000 == 9999:
                    log_writer.flushLog()

        if log_metrics:
            log_writer.close()

    def generate(self, from_chk = True, batch_size=64, save_img = True, r_dir = "/input", w_dir = "/output", step = None):

        if from_chk:
            saver = tf.train.Saver()
            ckpt  = tf.train.get_checkpoint_state(r_dir)
            if ckpt:
                print ("Restoring Model from : " + ckpt.model_checkpoint_path.replace("/output", r_dir))
                saver.restore(self.sess, ckpt.model_checkpoint_path.replace("/output", r_dir))
                print("Model restored.")


        gen_data = {"images":[], "clabels":[], "glabels":[], "dlabels":[]}
        if save_img:
            gen_image = np.zeros((self.image_dim * batch_size, self.image_dim * self.label_dim, self.channel))


        class_one_hot = np.identity(10, int)
        for i in range(self.label_dim):
            batch_noise = self.noise_data(batch_size, self.noise_dim)

            # Generate one-hot vectors as class condition
            label_f = np.zeros((batch_size, self.label_dim))
            label_f[:, i] = 1

            output, predictc, predictd = self.sess.run([tf.transpose(self.fake_data, (0, 2, 3, 1)), 
                                                      self.cat_fake, self.disc_fake], 
                                            feed_dict={self.fake_input: batch_noise, self.fake_label: label_f})


            if save_img:
                dim = self.image_dim
                gen_image[:, dim*i:dim*(i+1), :] = np.reshape(output, (batch_size * dim, dim, self.channel))

            for j in range(batch_size):
                gen_data["images"].append(output[j])
                gen_data["clabels"].append(np.argmax(predictc[j]))
                gen_data["glabels"].append(class_one_hot[i])
                gen_data["dlabels"].append(predictd[j])

        if save_img:
            gen_image = ((gen_image / 2 + 0.5)*255).astype(np.uint8)
            cv2.imwrite(os.path.join(w_dir, 'sample-{}.png'.format(step)), gen_image)

        dump_file = "gen_img.p"
        if step is not None:
            dump_file = "gen_img-{}.p".format(step)
        pickle.dump(gen_data, open(os.path.join(w_dir, dump_file), "wb" ) )

    def test(self, top_n = 5, n_samples = 5, r_dir = '/input', p_dir = '/input', real = True, gfilename = "gen_img.p"):
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(r_dir)
        if ckpt:
            print ("Restoring Model from : " + ckpt.model_checkpoint_path.replace("/output", r_dir))
            saver.restore(self.sess, ckpt.model_checkpoint_path.replace("/output", r_dir))
            print("Model restored.")

        if real:
            test_images, test_labels = self.input_data("test")
        else:
            print("Testing the Generated")
            test_images, test_clabels, test_labels = self.input_data.generated(r_dir = p_dir, filename = gfilename)

        accC = None

        random_data = list(zip(*random.sample(list(zip(test_images, test_labels)), n_samples)))

        top_predictions = self.sess.run(tf.nn.top_k(tf.nn.softmax(self.cat_real), top_n), feed_dict={self.real_data: random_data[0]})

        predictions = self.sess.run(tf.nn.softmax(self.cat_real), feed_dict={self.real_data: test_images})
        correct_pred = np.equal(np.argmax(predictions, 1), np.argmax(test_labels, 1)).astype(np.float32)
        acc = np.mean(correct_pred)

        if not real:
            correct_pred = np.equal(np.argmax(predictions, 1), np.argmax(test_clabels, 1)).astype(np.float32)
            accC = np.mean(correct_pred)

        return np.array(random_data[0]), np.array(random_data[1]), top_predictions, accC, acc

