
import os
import tensorflow as tf
import time

from models import mnist_model as model
from utils import log
from utils import sampler as data

class DeepNet_MNIST(object):
    def __init__(self, model, input_data):
        self.model     = model
        self.image_dim = 32
        self.label_dim = 10
        self.channel   = 1

        self.input_data = input_data

        self.data  = tf.placeholder(dtype=tf.float32, shape=(None, self.channel, self.image_dim, self.image_dim))
        self.label = tf.placeholder(dtype=tf.float32, shape=(None, self.label_dim))

        # Classifier - Model
        self.logits = self.model(self.data, reuse=False)
        self.logits = tf.identity(self.logits, name='logits')

        # Evaluation Metric - Loss
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.label))

        # Optimization
        self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)

        # Accuracy
        correct_pred = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.label, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')
        self.sess = tf.Session()

    def train(self, batch_size=128, epochs=200, log_metrics=False, w_dir = "/output", save_chkp=True):
        self.sess.run(tf.global_variables_initializer())
        start_time = time.time()

        saver = tf.train.Saver()

        vimages, vlabels = self.input_data("validation")
        vfeed_dict = {self.data: vimages, self.label: vlabels}
        plot_data = {"train_acc": 0, "valid_acc": 0, "train_loss": 0, "val_loss":0}

        if log_metrics:
            log_writer = log.LogWriter(w_dir, "mnist_train.csv")
            log_writer(plot_data)

        for i in range(epochs):
            train_images, train_labels = self.input_data("train", batch_size)
            tfeed_dict = {self.data: train_images, self.label: train_labels}

            _, tloss, train_acc = self.sess.run([self.optimizer, self.loss, self.accuracy], feed_dict=tfeed_dict)
            _, vloss, valid_acc = self.sess.run([self.optimizer, self.loss, self.accuracy], feed_dict=vfeed_dict)

            plot_data = {"train_acc": 0, "valid_acc": 0, "train_loss": 0, "val_loss":0}

            plot_data["train_acc"]  = train_acc
            plot_data["valid_acc"]  = valid_acc
            plot_data["train_loss"] = tloss
            plot_data["val_loss"]   = vloss

            if log_metrics:
                log_writer.writeLog(plot_data)

            log_data  = 'Iter: {} '.format(i) 
            log_data += 'Time: {:>5.4f} '.format(time.time() - start_time)
            log_data += 'Train Loss: {:>6.4f} '.format(plot_data["train_loss"]) 
            log_data += 'Valid Loss: {:>6.4f} '.format(plot_data["val_loss"]) 
            log_data += 'Train Acc: {:>6.4f} '.format(plot_data["train_acc"]) 
            log_data += 'Valid Acc: {:>6.4f} '.format(plot_data["valid_acc"]) 

            print(log_data)

        if log_metrics:
            log_writer.close()

        if save_chkp:
            # Save model
            save_path = saver.save(self.sess, os.path.join(w_dir, "mnist_model.ckpt"))
            print("Model saved in file: %s" % save_path)

    def test(self, top_n = 5, n_samples = 5, r_dir = '/input', real = True, p_dir="/input", gfilename = "gen_img.p"):
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

        tfeed_dict = {self.data: test_images, self.label: test_labels}

        predictions, acc = self.sess.run([tf.nn.softmax(self.logits), self.accuracy], feed_dict=tfeed_dict)
        if not real:
            predictions, accC = self.sess.run([tf.nn.softmax(self.logits), self.accuracy], 
                                  feed_dict={self.data: test_images, self.label: test_clabels})

        return np.array(random_data[0]), np.array(random_data[1]), top_predictions, accC, acc

