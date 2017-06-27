import os, sys
sys.path.append(os.getcwd())

import argparse

import cond_wgan
import mnist
import visuals as vs
from models import gan_model as model
from models import mnist_model as mmodel

from utils import log
from utils import sampler as data


def rmnist(r_dir='/input', p_dir = "/input", w_dir = '/output', epochs = 200, train = True, test = True, real = True, gfilename = "gen_img.p"):
    input_data = data.DataSampler()
    dconv_model= mmodel.Classifier('mnist_classifier', 32, 'NCHW', 10)
    deepnet = mnist.DeepNet_MNIST(dconv_model, input_data)
    if args.train:
        deepnet.train(log_metrics=True, w_dir = w_dir, epochs=epochs)
    if args.test:
        test_images, test_labels, top_predictions, accuracy, accuracyG = deepnet.test(5, 5, r_dir, real, p_dir, gfilename)
        vs.display_predictions(test_images, test_labels, top_predictions, 5, w_dir)
        print('Testing Accuracy: {}\n'.format(accuracy * 100.0))
        if accuracyG is not None:
            print('Testing Generator Accuracy: {}\n'.format(accuracyG * 100.0))

def cwgan(r_dir='/input', w_dir = '/output', epochs = 20000, train = True, test = True, gen = True, save_img = True, p_dir = "/input", 
                  gfilename = "gen_img.p"):
    disc_model = model.Discriminator(model_dim = 64, data_format = 'NCHW')
    gen_model  = model.Generator(noise_dim = 128, model_dim = 64, channel = 1, data_format = 'NCHW')
    dconv_model= mmodel.Classifier('classifier', model_dim = 32, data_format = 'NCHW', label_dim = 10)
    input_data = data.DataSampler()
    noise      = data.NoiseSampler()
    wgan       = cond_wgan.WassersteinGAN(gen_model, disc_model, dconv_model, input_data, noise)

    if train:
        wgan.train(log_metrics=True, w_dir = w_dir, epochs=epochs)
    if gen and not test:
        wgan.generate(batch_size=128, r_dir = r_dir, w_dir = w_dir, save_img = save_img)
    if test:
        test_images, test_labels, top_predictions, accuracy, accuracyG = wgan.test(top_n = 5, r_dir = r_dir, p_dir = p_dir, real = not gen, gfilename = gfilename)
        vs.display_predictions(test_images, test_labels, top_predictions, 5, w_dir)
        print('Testing Accuracy: {}\n'.format(accuracy * 100.0))
        if accuracyG is not None:
            print('Testing Generator Accuracy: {}\n'.format(accuracyG * 100.0))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('-m', type=str, default='mnist')
    parser.add_argument('-rd', type=str, default='/input')
    parser.add_argument('-wd', type=str, default='/output')
    parser.add_argument('-pd', type=str, default='/input')
    parser.add_argument('-pf', type=str, default='None')
    parser.add_argument('-train', type=bool, default='')
    parser.add_argument('-test', type=bool, default='')
    parser.add_argument('-gen', type=bool, default='')
    parser.add_argument('-e', type=int, default='200')
    parser.add_argument('-sim', type=bool, default='')

    args = parser.parse_args()
    filename = "gen_img-{}.p".format(args.pf)
    if args.pf == "None":
        filename = "gen_img.p".format(args.pf)

    if args.m == "mnist":
        rmnist(r_dir=args.rd, w_dir = args.wd, p_dir = args.pd, epochs = args.e, train = args.train, real = not args.gen, 
                  test = args.test, gfilename = filename)
    else:
        cwgan(r_dir=args.rd, w_dir = args.wd, epochs = args.e, train = args.train, test = args.test, 
                gen = args.gen, save_img = args.sim, p_dir = args.pd, gfilename = filename)


