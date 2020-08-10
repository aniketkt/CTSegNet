# -*- coding: utf-8 -*-
"""

"""
import os
import numpy as np
import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
import h5py
import sys
from tensorflow import keras
from IPython.display import clear_output
import random
import pandas as pd
from multiprocessing import cpu_count
from ct_segnet.data_utils.data_io import Parallelize
from ct_segnet.stats import *#ROC, calc_jac_acc, calc_dice_coeff, fidelity
from ct_segnet.data_utils.data_augmenter import run_augmenter

def _norm(y):
    if (y.max() != 1.0) | (y.min() != 0.0):
        y = (y - np.min(y)) / (np.max(y) - np.min(y))
    return y.astype(np.uint8)

def data_generator(X, Y, batch_size, \
                   check_norm = False, \
                   augmenter_todo = None, \
                   min_SNR = 2.0, \
                   inplace = False, nprocs = 1):
    """
    Generator that yields randomly sampled data pairs of size batch_size. X, Y are DataFile instance pairs of train / test / validation data.  
    
    Parameters
    ----------
    X: DataFile
        Input images  
    
    Y: DataFile
        Ground truth segmentation  
        
    check_norm : bool
        Ensure that segmentation labels are 0 and 1 only. Should be False if your ground truth data contains ignored pixels.  
        
    augmenter_todo  : list
        list of strings from "rotate", "flip", "gaussian noise", "invert intensity"
    
    min_SNR : float
        minimum allowable signal-to-noise ratio (SNR)
    
    Returns
    -------
    tuple
        X, Y numpy arrays each of shape (n_batch, ny, ny)  
    
    """
    while True:
        idxs = sorted(random.sample(range(X.d_shape[0]), batch_size))
        x = X.read_sequence(idxs)
        y = Y.read_sequence(idxs)
        
        if check_norm:
            y = _norm(y)
        
        if augmenter_todo is not None:
            x, y = run_augmenter(x, y, \
                                 min_SNR = min_SNR, \
                                 to_do = augmenter_todo, \
                                 inplace = inplace, \
                                 nprocs = nprocs)

        yield (x[...,np.newaxis], y[...,np.newaxis])

class Logger(keras.callbacks.Callback):
    """  
    An instance of Logger can be passed to keras model.fit to log stuff at epochs.  
    
    Parameters
    ----------
    model_paths : dict
        with keys = ["name", "history", "file"]  
    
    Xtest : DataFile
        input images from test data for calculating model accuracy  
    
    Ytest : DataFile
        ground truth segmentation map from test data  
    
    df_prev : pandas.DataFrame
        If retraining, pass previous epochs data  
    
    N : int
        autosave frequency (in epochs)  
    
    n_test : int
        number of test image pairs to be sampled every epoch  
    
    """
    def __init__(self, Xtest, Ytest, model_paths, N,\
                 df_prev = None, n_test = None, \
                 check_norm = False, augmenter_todo = None,):
        
        self.test_dg = data_generator(Xtest, Ytest, n_test, \
                                      check_norm = check_norm, \
                                      augmenter_todo = augmenter_todo)
        self.N = N
        self.df_prev = df_prev
        self.model_name = model_paths['name']
        self.model_history = model_paths['history']
        self.model_file = model_paths['file']
            
    
    def on_train_begin(self, logs={}):
        
        if self.df_prev is not None:
            next_epoch = self.df_prev['epoch'].tolist()[-1] + 1
            print("\n\nRestarting training from epoch # %i..."%next_epoch)

        # INITIALIZE VECTORS
        self.i = 1 if self.df_prev is None else next_epoch
        self.x = [] if self.df_prev is None else self.df_prev['epoch'].tolist()
        self.losses = [] if self.df_prev is None else self.df_prev['loss'].tolist()
        self.val_losses = [] if self.df_prev is None else self.df_prev['val_loss'].tolist()
        self.test_losses = [] if self.df_prev is None else self.df_prev['test_loss'].tolist()
        self.metric = [] if self.df_prev is None else self.df_prev['metric'].tolist()
        self.acc_zeros = [] if self.df_prev is None else self.df_prev['acc_zeros'].tolist()
        self.acc_ones = [] if self.df_prev is None else self.df_prev['acc_ones'].tolist()

        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        
        self.fig, self.ax = plt.subplots(1,2, figsize = (12,5))
        self.ax[0].set_xlabel('epoch')
        self.ax[0].set_ylabel('loss')
        self.ax[1].set_xlabel('epoch')
        self.ax[1].set_ylabel('metrics')
        

        # APPEND DATA
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        x_test, y_test = next(self.test_dg)
        evaluation = self.model.evaluate(x = x_test, y = y_test, verbose = 0)
        if type(evaluation) is list:
            if len(evaluation) == 4:
                self.test_losses.append(evaluation[0])
                self.metric.append(evaluation[1])
                self.acc_zeros.append(evaluation[2])
                self.acc_ones.append(evaluation[3])
            else:
                raise ValueError("\n number of metrics provided does not match plotter's configuration.")
        else:
            raise ValueError("\n expecting metric definitions but none found.")
        
        print("val_loss on test data: %.4f"%self.test_losses[-1])
        print("metric evaluated on test data: %.4f"%self.metric[-1])
        print("acc_zeros: %.2f, acc_ones: %.2f"%(self.acc_zeros[-1], self.acc_ones[-1]))
        
        clear_output(wait=True)
        self.ax[0].plot(self.x, self.losses, label="loss")
        self.ax[0].plot(self.x, self.val_losses, label="val_loss")
        self.ax[0].plot(self.x, self.test_losses, label = "test_data val_loss")
        self.ax[0].legend(loc = 'upper right')
        
        
        self.ax[1].plot(self.x, self.acc_zeros, label = 'acc_zeros')
        self.ax[1].plot(self.x, self.acc_ones, label = 'acc_ones')
        self.ax[1].plot(self.x, self.metric, '--k', label = "IoU")
        self.ax[1].legend(loc = 'upper right')
        
#         plt.show()
        plt.savefig(os.path.join(self.model_history, 'Loss_plot_' + self.model_name + '.png'))
        plt.close()     
        
        if self.i % self.N == 0:

            # Save loss data in .csv file
            df = pd.DataFrame()
            df['epoch'] = self.x
            df['loss'] = self.losses
            df['val_loss'] = self.val_losses
            df['test_loss'] = self.test_losses
            df['metric'] = self.metric
            df['acc_zeros'] = self.acc_zeros
            df['acc_ones'] = self.acc_ones
            df.set_index('epoch')
            df.to_csv(os.path.join(self.model_history,self.model_name+".csv"))

        
            if self.test_losses[-1] < 1.5*self.test_losses[-self.N]:
                print("Saving model at epoch # %i..."%self.i)
                self.model.save(self.model_file)
                self.model.save_weights(os.path.join(self.model_history, 'weights_' + self.model_name + '.h5'))
            else:
                print("Terminate Training: Overfitting detected from increasing test loss...")
                # self.model.stop_training = True
        
        self.i += 1


    def on_train_end(self, logs = {}):
        
        self.logs.append(logs)
        
        

def save_datasnaps(dg, model_history, n_imgs = 20):
    """
    """
    if not os.path.exists(os.path.join(model_history,"data_snaps")):
        os.makedirs(os.path.join(model_history,"data_snaps"))
        
        
    # Load and save random train data images
    x, y = next(dg)
    x, y = x[...,0], y[...,0]
    for jj in x.shape[0]:
        fig, ax = plt.subplots(1,2, figsize = (6,3))
        ax[0].axis('off')
        ax[0].imshow(y[jj])
        ax[1].axis('off')
        ax[1].imshow(x[jj])
        fig.suptitle("MIN: %.2f, MAX: %.2f, MEAN: %.2f"%(np.min(x[jj]), np.max(x[jj]), np.mean(x[jj])))
        plt.savefig(os.path.join(model_history,"data_snaps",'randomdata_%5d.png'%jj))
        plt.close()


    return






    
    

def save_results(dg, model_results, segmenter):
    """Save some results on test images into a folder in the path to model repo
    """
    x_test, y_test = next(dg)
    y_pred = segmenter.predict(x_test)
    y_pred = np.round(y_pred)
    
    x_test, y_test, y_pred = x_test[...,0], y_test[...,0], y_pred[...,0]
    
    if not os.path.exists(os.path.join(model_results,"data_snaps")):
        os.makedirs(os.path.join(model_results,"data_snaps"))
    
    for ii in range(x_test.shape[0]):
            
            fig, ax = plt.subplots(1,3, figsize = (9,3))
            ax[0].axis('off')
            ax[0].imshow(x_test[ii])
            ax[1].axis('off')
            ax[1].imshow(y_test[ii])
            ax[2].axis('off')
            ax[2].imshow(y_pred[ii])
            
            #jac_acc = (np.sum(y_pred[ii]*y_test[ii]) + 1) / (np.sum(y_pred[ii]) + np.sum(y_test[ii]) - np.sum(y_pred[ii]*y_test[ii]) + 1)
            test_loss, jac_acc, acc_zeros, acc_ones = segmenter.evaluate(x = x_test[ii][np.newaxis,...,np.newaxis], y = y_test[ii][np.newaxis,...,np.newaxis], verbose = 0)
            
            fig.suptitle("IMG_MEAN: %.2f, JAC_ACC: %.2f, LOSS: %.2f"%(np.mean(x_test[ii]),jac_acc, test_loss))
            plt.savefig(os.path.join(model_results,"data_snaps",'snap_%05d.png'%ii))
            plt.close()

    return


















