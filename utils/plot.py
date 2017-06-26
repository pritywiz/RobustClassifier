import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pandas as pd
import os

class PlotData(object):
    def __init__(self, save = False, from_file = False, show = True):
        self.show       = show
        self.save       = save
        self.from_file  = from_file
        self.rfilename  = ""
        self.wrfilename = ""
        self.title = ""
        self.plot_data  = None

    def __call__(self, title, rdir="/input", rfilename="plot.csv", wdir="/output", wfilename="plot.png"):
        if self.from_file:
            self.rfilename = os.path.join(rdir, rfilename)
            self.plot_data = pd.read_csv(self.rfilename)

        if self.save:
            self.wfilename = os.path.join(wdir, wfilename)
        self.title = title
        self.plot_stats()

    def plot_stats(self):
        axs = []
        if len(self.title["title"]) == 2:
            fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))
        else: 
            axs = [None, None, None, None]
            fig, ((axs[0]), (axs[1]), (axs[2]), (axs[3])) = plt.subplots(nrows=4, ncols=1, figsize=(18, 28))

        for spidx in range(len(self.title["title"])):
            if self.save:
                print ("Plotting for {}".format(self.title["title"][spidx]))

            x_data_idx = self.title['data_key'][spidx]
            for didx in range(len(x_data_idx)):
                x_data = self.plot_data[x_data_idx[didx]]
                axs[spidx].plot(range(1, len(x_data) + 1), x_data)

            axs[spidx].set_title(self.title["title"][spidx])
            axs[spidx].set_ylabel(self.title["y_label"][spidx])
            axs[spidx].set_xlabel(self.title["x_label"][spidx])

            if "legend" in self.title:
                axs[spidx].legend(self.title["legend"][spidx], loc='best')

            axs[spidx].set_xticks(np.arange(-5, len(x_data) + 1), len(x_data)/10)

        if self.save:
            plt.savefig(self.wfilename)
        if self.show:
            plt.show()
        pass


