import os, shutil
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Lock


def multiplot(wave, flux, model, N, xlbl, ylbl):
    di = len(wave)//N
    fig, axes = plt.subplots(N, 1)
    for i in range(N):
        i1 = i*di
        i2 = (i+1)*di

        axes[i].plot(wave[i1:i2], flux[i1:i2], label='Observation')
        axes[i].plot(wave[i1:i2], model[i1:i2], label='Model')

        if ylbl!=None: axes[i].set_ylabel(ylbl)
        if i==N-1:
            if xlbl!=None: axes[i].set_xlabel(xlbl)
            axes[i].legend()

class FitLogger:

    def __init__(self, log_dir, delete_old=False):
        if delete_old: shutil.rmtree(log_dir, ignore_errors=True)
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self.log_file_name = 'LOG'
        self.N_subplots = 5
        self.figure_width = 20
        self.figure_height = 10
        self.dpi = 300
        self.lock = Lock()

    def save_plot(self, wave, flux, model, name):
        N = self.N_subplots

        multiplot(wave, flux, model, N, 'Wavelength [A]', 'Flux')
        
        fig = plt.gcf()
        fig.set_size_inches(self.figure_width, self.figure_height)
        plt.tight_layout()

        path = os.path.join(self.log_dir, name)
        fig.savefig(path + '.png', dpi=self.dpi)
        fig.savefig(path + '.pdf', dpi=self.dpi)
        
        fig.clf()

    def add_record(self, text):
        self.lock.acquire()
        path = os.path.join(self.log_dir, self.log_file_name)
        with open(path, 'a') as f:
            f.write(text)
            f.write('\n')
        self.lock.release()







