# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 00:59:58 2019

@author: atekawade
"""

import numpy as np
import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
from ImageStackPy import ImageProcessing as IP
from ImageStackPy import Img_Viewer as VIEW
import h5py
import os
import sys
import keras
from IPython.display import clear_output
import random
import pandas as pd
from multiprocessing import cpu_count


class Logger(keras.callbacks.Callback):

    def __init__(self, x_val, y_val, model_paths, N, df_prev = None):
        self.x_val = x_val
        self.y_val = y_val
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
        evaluation = self.model.evaluate(x = self.x_val[...,np.newaxis], y = self.y_val[...,np.newaxis])
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
        
        plt.show()
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
        
        

def save_datasnaps(x_train, y_train, model_history):
    
    if not os.path.exists(os.path.join(model_history,"data_snaps")):
        os.makedirs(os.path.join(model_history,"data_snaps"))
        
        
    # Load and save random train data images
    for ii in range(100):
        jj = random.randrange(y_train.shape[0])
        fig, ax = plt.subplots(1,2, figsize = (6,3))
        ax[0].axis('off')
        ax[0].imshow(y_train[jj])
        ax[1].axis('off')
        ax[1].imshow(x_train[jj])
        fig.suptitle("MIN: %.2f, MAX: %.2f, MEAN: %.2f"%(np.min(x_train[jj]), np.max(x_train[jj]), np.mean(x_train[jj])))
        plt.savefig(os.path.join(model_history,"data_snaps",'randomdata_%5d.png'%jj))
        plt.close()

    for ii in range(100,200):
            
            fig, ax = plt.subplots(1,2, figsize = (6,3))
            ax[0].axis('off')
            ax[0].imshow(y_train[ii])
            ax[1].axis('off')
            ax[1].imshow(x_train[ii])
            fig.suptitle("MIN: %.2f, MAX: %.2f, MEAN: %.2f"%(np.min(x_train[ii]), np.max(x_train[ii]), np.mean(x_train[ii])))
            plt.savefig(os.path.join(model_history,"data_snaps",'serialdata_%5d.png'%ii))
            plt.close()

    return




def ROC(thresh, y_true = None, y_pred = None):
    
    y_p = np.zeros_like(y_pred)
    y_p[y_pred > thresh] = 1
    y_true = np.copy(y_true)

    TN = np.sum((1-y_true)*(1-y_p)).astype(np.float32)
    
    FP = np.sum((1-y_true)*y_p).astype(np.float32)
    
    TNR = TN / (TN + FP)
    FPR = 1 - TNR
    
    TP = np.sum(y_true*y_p).astype(np.float32)
    
    FN = np.sum(y_true*(1-y_p)).astype(np.float32)
    
    TPR = TP / (TP +  FN)
    
    return (FPR, TPR)


def calc_jac_acc(y_true, y_pred):
    
    y_pred = np.round(np.copy(y_pred))
    jac_acc = (np.sum(y_pred*y_true) + 1) / (np.sum(y_pred) + np.sum(y_true) - np.sum(y_pred*y_true) + 1)
    return jac_acc



#def calc_jac_acc(y_true, y_pred):
#    # Pixel-wise accuracy
#    y_pred = np.round(np.copy(y_pred))
#    jac_acc = (np.sum(y_pred*y_true) + np.sum((1-y_pred)*(1-y_true))) / np.size(y_true)
#    return jac_acc


def fidelity(y_true, y_pred, tolerance = 0.95):
    

    XY = [(y_true[ii], y_pred[ii]) for ii in range(y_true.shape[0])]
    del y_true
    del y_pred
    
    jac_acc = np.asarray(IP.Parallelize(XY, calc_jac_acc, procs = cpu_count()))
    
    mean_IoU = np.mean(jac_acc)

    jac_fid = np.zeros_like(jac_acc)
    jac_fid[jac_acc > tolerance] = 1
    jac_fid = np.sum(jac_fid).astype(np.float32) / np.size(jac_acc)
    
    
    return jac_fid, mean_IoU, jac_acc
    
    

def save_results(x_val, y_val, y_pred, model_results, segmenter):
    
    if not os.path.exists(os.path.join(model_results,"data_snaps")):
        os.makedirs(os.path.join(model_results,"data_snaps"))
        
    for ii in range(x_val.shape[0]):
            
            fig, ax = plt.subplots(1,3, figsize = (9,3))
            ax[0].axis('off')
            ax[0].imshow(x_val[ii])
            ax[1].axis('off')
            ax[1].imshow(y_val[ii])
            ax[2].axis('off')
            ax[2].imshow(y_pred[ii])
            
            #jac_acc = (np.sum(y_pred[ii]*y_val[ii]) + 1) / (np.sum(y_pred[ii]) + np.sum(y_val[ii]) - np.sum(y_pred[ii]*y_val[ii]) + 1)
            test_loss, jac_acc, acc_zeros, acc_ones = segmenter.evaluate(x = x_val[ii][np.newaxis,...,np.newaxis], y = y_val[ii][np.newaxis,...,np.newaxis])
            
            fig.suptitle("IMG_MEAN: %.2f, JAC_ACC: %.2f, LOSS: %.2f"%(np.mean(x_val[ii]),jac_acc, test_loss))
            plt.savefig(os.path.join(model_results,"data_snaps",'snap_%05d.png'%ii))
            plt.close()

    return


def save_results_2(x_val, y_val, y_pred, model_results, jac_acc):
    
    if not os.path.exists(os.path.join(model_results,"data_snaps")):
        os.makedirs(os.path.join(model_results,"data_snaps"))
        
    for ii in range(x_val.shape[0]):
            
            fig, ax = plt.subplots(1,3, figsize = (9,3))
            ax[0].axis('off')
            ax[0].imshow(x_val[ii], cmap = 'gray')
            ax[1].axis('off')
            ax[1].imshow(y_val[ii])
            ax[2].axis('off')
            ax[2].imshow(y_pred[ii])
            
            
            fig.suptitle("IMG_MEAN: %.2f, JAC_ACC: %.2f"%(np.mean(x_val[ii]),jac_acc[ii]))
            plt.savefig(os.path.join(model_results,"data_snaps",'snap_%05d.png'%ii))
            plt.close()

    return




















