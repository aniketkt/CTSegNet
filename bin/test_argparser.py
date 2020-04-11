#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 16:14:21 2019

@author: atekawade
This script reads hyperparameters from a .config file to define an fCNN model, then trains it on data for which filepaths are provided.
"""
import configargparse

import numpy as np
import os
import sys
import matplotlib as mpl

def make_tuples(inp):
    if len(inp) == 1:
        return (inp[0], inp[0])
    else:
        return [(_inp, _inp) for _inp in inp]


def main():

    mpl.use('Agg')
    # Inputs from argparser
    kern_size = make_tuples(kern_size)
    kern_size_upconv = make_tuples(kern_size_upconv)
    pool_size =  make_tuples(pool_size)
    
    
    return


if __name__ == "__main__":

    import configargparse

    parser = configargparse.ArgParser()
    
    # Train config file
    parser.add('-t', '--config-train', required=True, is_config_file=True, help='training config file path')
    parser.add('--model_name', required = True, type = str, help = 'str; a unique name for the model')
    parser.add('--rebuild', required = True, type = bool, help = 'bool; True to rebuild model from model config file, False to retrain existing')
    parser.add('--initialize_from', required = False, default = None, type = str, help = 'str or None, path to weights (.h5) file')
    parser.add('--n_epochs', required = True, type = int, help = 'int; number of epochs')
    parser.add('--batch_size', required = True, type = int, help = 'int; batch size')
    parser.add('--autosave_freq', required = False, default = 10, type = int, help = 'int; autosave model after these many epochs')
    parser.add('--skip_fac', required = False, default = 1, type = int, help = 'int; step value when slicing through train / test data')
    parser.add('--model_path', required = True, type = str, help = 'str; path to parent directory containing all models (model repository)')
    parser.add('--data_path', required = True, type = str, help = 'str; path to parent directory containing train / test data')
    parser.add('--train_fname', required = True, type = str, help = 'str; filename of .hdf5 file as training set')
    parser.add('--test_fname', required = True, type = str, help = 'str; filename of .hdf5 file as testing set')
    parser.add('--data_key', required = True, type = str, help = 'str; brief description of datasets')
    
    

    # Model config file
    parser.add('-m', '--config-model', required=False, is_config_file=True, help='model config file path (optional)')
    parser.add('--model_size', required = False, default = 512, type = int, help = 'int; input/output image size e.g. 512 for 512x512 image')
    parser.add('--loss_def', required = False, default = 'focal_loss', type = str, help = 'str; loss function definition')
    parser.add('--activation', required = False, type = str, default = 'lrelu', help = 'str; activation of conv. layers')    
    parser.add('--is_batch_norm', required = False, default = True, help = 'bool; True to add batch normalization')
    parser.add('--stdinput', required = False, default = True, help = 'bool; True to normalize input image intensity')
    parser.add('--n_pools', required = False, default = 3, help = 'int; number of pooling layers')
    parser.add('--n_depth', required = False, type = int, default = [32, 64, 128, 256], help = 'int; list of filter sizes', nargs = '+')
    parser.add('--kern_size', required = False, default = 3, type = int, help = 'int or list of conv layer kernel sizes', nargs = '+')
    parser.add('--kern_size_upconv', required = False, default = 2, type = int, help = 'int or list of upconv layer kernel sizes', nargs = '+')   
    parser.add('--pool_size', required = False, default = [2,4,2], type = int, help = 'int or list of max pool sizes', nargs = '+')    
    parser.add('--dropout_level', required = False, default = 1.0, type = float, help = 'float; dropout scaling factor')
    
    args = parser.parse_args()


#     globals().update(vars(args))

#     print("----------")
#     print(parser.format_help())
#     print("----------")
#     print(parser.format_values())    # useful for logging where different settings came from    
    
    args_summary = parser.format_values()
#     print(args)
    main()
    

    
