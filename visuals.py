import numpy as np
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils import plot

def visualize_mnist(images, labels, w_dir= "/output"):
    num_classes = labels.shape[1]
    num_images = 5
    fig = plt.figure(figsize=(16,8))
    for i in range(num_classes):
        index_cur_label = np.where(labels[:, i]==1)[0]
        rand_index = np.random.choice(index_cur_label, size=(num_images))
        image = images[rand_index]
        for j in range(num_images):
            ax = fig.add_subplot(num_images, num_classes, i*num_images + (j+1), xticks=[], yticks=[])
            ax.margins(0.1, 0.1)
            ax.set_title(i)
            plt.imshow(image[j][0], cmap='gray')
    plt.savefig(os.path.join(w_dir, "mnist_data.png"))
    plt.show()

def plot_mnist_train(rdir = "/mnist", w_dir= "/output"):
    title = {"title"   : ["Model Accuracy", "Model Loss"],
             "legend"  : [["Training", "Validation"], ["Training", "Validation"]], 
             "y_label" : ["Accuracy", "Loss"],
             "x_label" : ["Epoch", "Epoch"],
             "data_key": [['train_acc', 'valid_acc'], ['train_loss', 'val_loss']]}
    plt_fig = plot.PlotData(save = True, from_file = True)
    plt_fig(title, rdir=rdir, rfilename="mnist_train.csv", wdir = w_dir, wfilename = "mnist_train.png")

def plot_wgan_train(rdir = "/mnist", w_dir= "/output"):
    title = {"title"   : ["Classifier", "Classifier Fake", "Discriminator", "Generator"],
             "y_label" : ["Loss","Loss", "Loss", "Loss"],
             "x_label" : ["Epoch", "Epoch", "Epoch", "Epoch"],
             "data_key": [['c_loss_r'], ['c_loss_f'], ['d_loss'], ['g_loss']]}
    plt_fig = plot.PlotData(save = True, from_file = True)
    plt_fig(title, rdir=rdir, rfilename="wgan_train.csv", wdir = w_dir, wfilename = "wgan_train.png")

def plot_labels_count(all_labels, data_names, labels_index = 10, w_dir = '/output'):
    
    width = 0.35
    fig, ax = plt.subplots()
    index = np.arange(labels_index)
    bar_width = 0.30
    opacity = 0.8
    left_pad = 0.1
    i = 0
    colors = []
    for each_labels in all_labels: 
        nhlabels = [np.where(v == 1)[0][0] for v in each_labels]
        _, labels_count = np.unique(nhlabels, return_counts=True)
        color=list(plt.rcParams['axes.prop_cycle'])[i+1]['color']
        ax.bar(index + bar_width * i, labels_count, bar_width, alpha=opacity, color=color, label=data_names[i])
        i+=1

    ax.set_xlabel('Labels')
    ax.set_ylabel('Labels Count')
    ax.set_title('Count of Labels')
    ax.set_xticks(index, (range(labels_index)))
    ax.legend()
 
    plt.tight_layout()
    plt.savefig(os.path.join(w_dir, "mnist_labels.png"))
    plt.show()


def plot_histogram(gdata, w_dir = '/output'):
    gdata = ((gdata / 2 + 0.5)*255).astype(np.uint8)
    plt.hist(gdata)
    plt.title("Gaussian Histogram of Data")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(w_dir, "gaus_hist.png"))
    plt.show()


def display_predictions(images, labels, top_predictions, top_n, w_dir):
    n_classes = labels.shape[1]

    fig, axies = plt.subplots(nrows=images.shape[0], ncols=2)
    fig.tight_layout()
    fig.suptitle('Softmax Predictions', fontsize=20, y=1.1)

    n_predictions = top_n
    margin = 0.05
    ind = np.arange(n_predictions)
    width = (1. - 2. * margin) / n_predictions
 
    for image_i, (image, label_id, pred_indicies, pred_values) in enumerate(zip(images, labels, top_predictions.indices, top_predictions.values)):
        correct_name = np.argmax(label_id)

        image = image.reshape(image.shape[1],image.shape[1])
        axies[image_i][0].imshow(image)
        axies[image_i][0].set_title(correct_name)
        axies[image_i][0].set_axis_off()

        axies[image_i][1].barh(ind + margin, pred_values[::-1], width)
        axies[image_i][1].set_yticks(ind + margin)
        axies[image_i][1].set_yticklabels(pred_indicies[::-1])
        axies[image_i][1].set_xticks([0, 0.5, 1.0])
    plt.savefig(os.path.join(w_dir, "predictios.png"))
    plt.show()


