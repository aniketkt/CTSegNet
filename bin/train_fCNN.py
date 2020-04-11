#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 16:14:21 2019

@author: atekawade
This script reads hyperparameters from a .config file to define an fCNN model, then trains it on data for which filepaths are provided.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
from contextlib import redirect_stdout
import configargparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import h5py
import sys

from tensorflow.keras.models import load_model
import pandas as pd
import time
import random

from ct_segnet.model_utils.models import build_Unet_flex as build_Unet
from ct_segnet.model_utils.losses import custom_objects_dict, alpha, gamma
from ct_segnet.train_utils import Logger
from ct_segnet.train_utils import save_datasnaps
from ct_segnet.train_utils import save_results, data_generator
from ct_segnet.data_utils import data_io

def _make_tuples(inp):
    if len(inp) == 1:
        return (inp[0], inp[0])
    else:
        return [(_inp, _inp) for _inp in inp]


def nullable_string(val):
    if val == "None":
        return None
    return val

    
def main(args_summary, **kwargs):

    mpl.use('Agg')
    
    
    # Inputs from argparser
    args.kern_size = _make_tuples(args.kern_size)
    args.kern_size_upconv = _make_tuples(args.kern_size_upconv)
    args.pool_size =  _make_tuples(args.pool_size)

    data_io.show_header()
    
    ############### LOAD OR REBUILD fCNN MODEL ####################
    model_file = os.path.join(args.model_path, args.model_name + ".hdf5")
    model_history = os.path.join(args.model_path,'history',args.model_name)
    if not os.path.exists(model_history): os.makedirs(model_history)

    if args.rebuild:
        if args.config_model is None:
            raise Exception("Model config file must be provided if rebuilding")
        
        img_shape = (args.model_size, args.model_size, 1)
        segmenter = build_Unet(img_shape, \
                               n_depth = args.n_depth,\
                               n_pools = args.n_pools,\
                               activation = args.activation,\
                               batch_norm = args.is_batch_norm,\
                               kern_size = args.kern_size,\
                               kern_size_upconv = args.kern_size_upconv,\
                               pool_size = args.pool_size,\
                               dropout_level = args.dropout_level,\
                               loss = args.loss_def,\
                               stdinput = args.stdinput)

        if args.initialize_from is not None:
            print("\nInitializing weights from file:\n%s"%args.initialize_from)
            segmenter.load_weights(args.initialize_from)
        
        segmenter.save(model_file)
        df_prev = None
    
    elif os.path.exists(os.path.join(model_history,args.model_name+".csv")):
        segmenter = load_model(model_file, custom_objects = custom_objects_dict)

        df_prev = pd.read_csv(os.path.join(model_history,args.model_name+".csv"))
    else:
        raise ValueError("Model history not available to retrain. Rebuild required.")

    ###### LOAD TRAINING AND TESTING DATA #############
    train_path = os.path.join(args.data_path, args.train_fname)
    X = data_io.DataFile(train_path, tiff = False, data_tag = 'X', VERBOSITY = 0)
    Y = data_io.DataFile(train_path, tiff = False, data_tag = 'Y', VERBOSITY = 0)
    
    # used to evaluate accuracy metrics during training
    test_path = os.path.join(args.data_path, args.test_fname)
    Xtest = data_io.DataFile(test_path, tiff = False, data_tag = 'X', VERBOSITY = 0)
    Ytest = data_io.DataFile(test_path, tiff = False, data_tag = 'Y', VERBOSITY = 0)
    
    print("\nTotal test data shape: " + str(Ytest.d_shape))
    print("Total training data shape: " + str(Y.d_shape))
        
#     save_datasnaps(data_generator(X, Y, args.n_save), model_history)
        
        

    ##### DO SOME LOGGING #####################
        
    rw = "w+" if args.rebuild else "a+"
    logfile = os.path.join(model_history,"README_"+args.model_name+".txt")
    with open(logfile, rw) as f:
        if args.rebuild:
            f.write("\n")
            with redirect_stdout(f):
                segmenter.summary()
        f.write("\nNew Entry\n")
        f.write("Training Started: " + time.ctime() + "\n")
        if not args.rebuild: args_summary = args_summary.split("Defaults")[0]
        f.write(args_summary)
        f.write("\nTotal train data size: %s"%(str(Y.d_shape)))        
        f.write("\nTotal test data size: %s"%(str(Ytest.d_shape))) 
        f.write("\n")
        
    ######### START TRAINING ##################
    model_paths = {'name' : args.model_name,\
                   'history' : model_history,\
                   'file' : model_file}
    
    logger = Logger(Xtest, Ytest, model_paths, args.autosave_freq, df_prev = df_prev, n_test = args.n_test)
    n_train = min(args.nmax_train, X.d_shape[0])
    t0 = time.time()
    
    try:

        if args.fit_generator:
            steps_per_epoch = int((1- args.validation_split)*n_train//args.batch_size)
            validation_steps = int(args.validation_split*n_train//args.batch_size)
            hist = segmenter.fit_generator(data_generator(X, Y, args.batch_size),\
                                           steps_per_epoch = steps_per_epoch,\
                                           validation_data = data_generator(X, Y, args.batch_size),\
                                           validation_steps = validation_steps,\
                                           epochs = args.n_epochs,\
                                           verbose = 2, \
                                           callbacks = [logger])   
        else:
            
            x_train, y_train = next(data_generator(X, Y, n_train))
            hist = segmenter.fit(x = x_train, y = y_train,\
                                 verbose = 2, initial_epoch = 0, validation_split = args.validation_split,\
                                 epochs = args.n_epochs, batch_size = args.batch_size, callbacks = [logger])   
            
        print("\nModel training complete")
        with open(logfile, "a+") as f:
            f.write("\nTraining Completed: " + time.ctime() + "\n")
            hours = (time.time() - t0)/(60*60.0)
            f.write("\nTotal time: %.4f hours\n"%(hours))
            f.write("\nEnd of Entry")
            print("\n Total time: %.4f hours"%(hours))
        segmenter.save(model_file)        

        ########### EVALUATED MODEL AND SAVE SOME TEST IMAGES ############
        print("\nSaving some test images...")
        model_results = os.path.join(args.model_path, 'history', args.model_name, 'testing')
        save_results(data_generator(Xtest, Ytest, args.n_save), model_results, segmenter)
        
    
    
    # Log keyboard interrupt exception
    except KeyboardInterrupt:
        print("\nModel training interrupted")
        with open(logfile, "a+") as f:
            f.write("\nTraining Interrupted after %i epochs: "%logger.i + time.ctime() + "\n")
            f.write("\nEnd of Entry")
    
    

    
    
    return
        
        


if __name__ == "__main__":

    parser = configargparse.ArgParser()    

    # Train config file
    parser.add('-t', '--config-train', required=True, is_config_file=True, help='training config file path')
    parser.add('--model_name', required = True, type = str, help = 'str; a unique name for the model')
    parser.add('--rebuild', required = False, type = bool, default = False, help = 'bool; True to rebuild model from model config file, False to retrain existing')
    parser.add('--fit_generator', required = False, type = bool, default = True, help = 'bool; True is optimal for low RAM usage')
    parser.add('--initialize_from', required = False, default = None, type = nullable_string, help = 'str or None, path to weights (.h5) file')
    parser.add('--n_epochs', required = True, type = int, help = 'int; number of epochs')
    parser.add('--batch_size', required = True, type = int, help = 'int; batch size')
    parser.add('--validation_split', required = False, default = 0.2, type = float, help = 'float, train / validation split ratio')
    parser.add('--n_test', required = False, default = 400, type = int, help = 'number of test data pairs')
    parser.add('--nmax_train', required = False, default = 10000, type = int, help = 'total number of data pairs for train + val')
    parser.add('--n_save', required = False, default = 50, type = int, help = 'number of data pair snapshots to save')
    parser.add('--autosave_freq', required = False, default = 10, type = int, help = 'int; autosave model after these many epochs')
    parser.add('--skip_fac', required = False, default = 1, type = int, help = 'int; step value when slicing through train / test data')
    parser.add('--model_path', required = True, type = str, help = 'str; path to parent directory containing all models (model repository)')
    parser.add('--data_path', required = True, type = str, help = 'str; path to parent directory containing train / test data')
    parser.add('--train_fname', required = True, type = str, help = 'str; filename of .hdf5 file as training set')
    parser.add('--test_fname', required = True, type = str, help = 'str; filename of .hdf5 file as testing set')
    parser.add('--data_key', required = True, type = str, help = 'str; brief description of datasets')
    
    

    # Model config file
    parser.add('-m', '--config-model', required=False, is_config_file=True, help='model config file path (optional)')
    parser.add('--model_size', required = False, default = 512, type = int, help = 'int; in/out image size e.g. 512 for 512x512 image')
    parser.add('--loss_def', required = False, default = 'focal_loss', type = str, help = 'str; loss function definition')
    parser.add('--activation', required = False, type = str, default = 'lrelu', help = 'str; activation of conv. layers')    
    parser.add('--is_batch_norm', required = False, default = True, help = 'bool; True to add batch normalization')
    parser.add('--stdinput', required = False, default = True, help = 'bool; True to normalize input image intensity')
    parser.add('--n_pools', required = False, type = int, default = 3, help = 'int; number of pooling layers')
    parser.add('--n_depth', required = False, type = int, default = [32, 64, 128, 256], help = 'int; list of filter sizes', nargs = '+')
    parser.add('--kern_size', required = False, type = int, default = [3], help = 'int or list of conv layer kernel sizes', nargs = '+')
    parser.add('--kern_size_upconv', required = False, type = int, default = [2], help = 'int or list of upconv layer kernel sizes', nargs = '+')   
    parser.add('--pool_size', required = False, type = int, default = [2,4,2], help = 'int or list of max pool sizes', nargs = '+')    
    parser.add('--dropout_level', required = False, type = float, default = 1.0, help = 'float; dropout scaling factor')

    args = parser.parse_args()
    args_summary = parser.format_values()
    main(args_summary, **vars(args))
    
