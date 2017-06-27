import cond_wgan
import mnist
from models import gan_model as model
from models import mnist_model as mmodel

from utils import sampler as data

def restore_mnist(input_data = None, r_dir = '/input'):
    dconv_model= mmodel.Classifier('mnist_classifier', 32, 'NCHW', 10)
    deepnet = mnist.DeepNet_MNIST(dconv_model, input_data)
    deepnet.restore(r_dir)
    return deepnet
 
def restore_wgan(input_data = None, r_dir = '/input'):
    disc_model = model.Discriminator(model_dim = 64, data_format = 'NCHW')
    gen_model  = model.Generator(noise_dim = 128, model_dim = 64, channel = 1, data_format = 'NCHW')
    dconv_model= mmodel.Classifier('classifier', model_dim = 32, data_format = 'NCHW', label_dim = 10)
    noise      = data.NoiseSampler()
    wgan       = cond_wgan.WassersteinGAN(gen_model, disc_model, dconv_model, input_data, noise)
    wgan.restore(r_dir)
    return wgan
 
def test_mnist(model = None, top_n = 5, r_dir = '/input'):
    test_images, test_labels, top_predictions, _, accuracy = model.test(top_n = top_n, r_dir = r_dir, real = True)
    return test_images, test_labels, top_predictions, accuracy

def test_wgan(model = None, top_n = 5, r_dir = '/input'):
    test_images, test_labels, top_predictions, _, accuracy = model.test(top_n = top_n, r_dir = r_dir, real = True)
    return test_images, test_labels, top_predictions, accuracy

def test_gen_mnist(model = None, top_n = 5, r_dir = '/input', p_dir = '/input', epochs = 30000):
    accuracy_epochs = {}

    for step in range(999, epochs, 1000):
        gfilename = "gen_img-{}.p".format(step)
        _, _, _, accuracy, accuracyG = model.test(top_n = top_n, r_dir = r_dir, p_dir = p_dir, real = False, gfilename = gfilename)
        accuracy_epochs[step] = accuracyG
    return accuracy_epochs

def test_gen_wgan(model = None, top_n = 5, r_dir = '/input', p_dir = '/input', epochs = 30000):
    accuracy_epochs = {}

    for step in range(999, epochs, 1000):
        gfilename = "gen_img-{}.p".format(step)
        _, _, _, accuracy, accuracyG = model.test(top_n = top_n, r_dir = r_dir, p_dir = p_dir, real = False, gfilename = gfilename)
        accuracy_epochs[step] = accuracyG
    return accuracy_epochs

