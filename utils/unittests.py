import sampler

def test_sampler(sampler):
    images, _ = sampler("train")
    assert list(images.shape) == [55000, 1, 32, 32], 'Incorrect Shape.  Found {} shape'.format(images.shape)

    images, _ = sampler("validation")
    assert list(images.shape) == [5000, 1, 32, 32], 'Incorrect Shape.  Found {} shape'.format(images.shape)
    images, _ = sampler("test")
    assert list(images.shape) == [10000, 1, 32, 32], 'Incorrect Shape.  Found {} shape'.format(images.shape)

    images, _ = sampler("train", 128)
    assert list(images.shape) == [128, 1, 32, 32], 'Incorrect Shape.  Found {} shape'.format(images.shape)
    print('Tests Passed')

    images, _ = sampler.perturbed(batch_size = 128)
    assert list(images.shape) == [128, 1, 32, 32], 'Incorrect Shape.  Found {} shape'.format(images.shape)
    print('Tests Passed')

import log
import csv

def test_log():
    try:
        log_writer = log.LogWriter("./", "test.csv")
        log_writer({"d_loss": 0.0, "g_loss": 0.0, "c_loss_r": 0.0, "c_loss_f":0.0})
        for i in range(5):
            inc = i * 4
            log_writer.writeLog({"d_loss": inc + 1, "g_loss": inc + 2, "c_loss_r": inc + 3, "c_loss_f": inc + 4})
        log_writer.close()
        with open("test.csv", 'r') as csvfile:
            spamreader = csv.reader(csvfile)
            for row in spamreader:
                print (row)
    except Exception as e:
        print("error {}".format(e))

import plot
def test_plot():
    title = {"title"   : ["Model Accuracy", "Model Loss"],
             "legend"  : [["Training", "Validation"], ["Training", "Validation"]], 
             "y_label" : ["Accuracy", "Loss"],
             "x_label" : ["Epoch", "Epoch"],
             "data_key": [['train_acc', 'valid_acc'], ['loss', 'val_loss']]}
    plt_fig = plot.PlotData(save = True, from_file = True)
    plt_fig(title, rdir="./", rfilename="test.csv", wdir="./", wfilename="plot.png")
    title = {"title"   : ["Classifier", "Classifier Fake", "Discriminator", "Discriminator"],
             "y_label" : ["Loss","Loss", "Loss", "Loss"],
             "x_label" : ["Epoch", "Epoch", "Epoch", "Epoch"],
             "data_key": [['train_acc'], ['valid_acc'], ['loss'], ['val_loss']]}
    plt_fig = plot.PlotData(save = True, from_file = True)
    plt_fig(title, rdir="./", rfilename="test.csv", wdir="./", wfilename="plot1.png")

if __name__ == '__main__':
    test_sampler(sampler.DataSampler())
    #test_log()
    #test_plot()

