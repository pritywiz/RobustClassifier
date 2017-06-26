import cond_wgan
import mnist
from models import gan_model as model
from models import mnist_model as mmodel

from utils import sampler as data
 
def test_mnist(m_dir="/mnist", real = True):
    input_data = data.DataSampler()
    dconv_model= model.Classifier(32, 'NCHW', 10)
    deepnet = mnist.DeepNet_MNIST(dconv_model, input_data)
    accuracy = deepnet.test(r_dir = m_dir, real = real)
    print('Testing Accuracy: {}\n'.format(accuracy * 100.0))

def test_wgan(top_n = 5, r_dir = '/input', p_dir = '/input', real = True, gfilename = "gen_img.p"):
    disc_model = model.Discriminator(model_dim = 64, data_format = 'NCHW')
    gen_model  = model.Generator(noise_dim = 128, model_dim = 64, channel = 1, data_format = 'NCHW')
    dconv_model= mmodel.Classifier(model_dim = 32, data_format = 'NCHW', label_dim = 10)
    input_data = data.DataSampler()
    noise      = data.NoiseSampler()
    wgan       = cond_wgan.WassersteinGAN(gen_model, disc_model, dconv_model, input_data, noise)

    predictions, accuracy = wgan.test(top_n = top_n, r_dir = r_dir, p_dir = p_dir, real = real, gfilename = gfilename)
    return predictions, accuracy


