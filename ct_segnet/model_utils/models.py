#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 14:12:21 2019

@author: atekawade
"""

import numpy as np
import matplotlib.pyplot as plt
from ImageStackPy import ImageProcessing as IP
from ImageStackPy import Img_Viewer as VIEW
import h5py
import os
import sys
from keras.backend import tf
import keras, keras.layers as L, keras.backend as K
import keras_utils
from sklearn.feature_extraction import image as feim
from . import losses

def insert_activation(tensor_in, activation):

    if activation == 'lrelu':
        tensor_out = L.LeakyReLU(alpha = 0.2)(tensor_in)
    else:
        tensor_out = L.Activation(activation)(tensor_in)
    
    return tensor_out
    


def conv_layer(tensor_in, n_filters, kern_size = None, activation = None, kern_init = 'he_normal', padding = 'same', dropout = 0.1, batch_norm = False):
    
    # layer # 1
    tensor_out = L.Conv2D(n_filters, kern_size, activation = None, kernel_initializer = kern_init, padding = padding)(tensor_in)
    
    if batch_norm:
        tensor_out = L.BatchNormalization(momentum = 0.9, epsilon = 1e-5)(tensor_out)
    
    tensor_out = insert_activation(tensor_out, activation)
    
    # Dropout
    if dropout is not None:
        tensor_out = L.Dropout(dropout)(tensor_out)
    
    # Layer # 2
    tensor_out = L.Conv2D(n_filters, kern_size, activation = None, kernel_initializer = kern_init, padding = padding)(tensor_out)

    if batch_norm:
            tensor_out = L.BatchNormalization(momentum = 0.9, epsilon = 1e-5)(tensor_out)
        
    tensor_out = insert_activation(tensor_out, activation)

    return tensor_out
    
def upconv_layer(tensor_in, concat_tensor, n_filters = None, activation = None, kern_size = None, strides = None, padding = 'same', batch_norm = False):
    
    if not n_filters:
        n_filters = int(tensor_in.shape[-1]) // 2
    
    tensor_out = L.Conv2DTranspose(n_filters, kern_size, strides = strides, padding = padding, activation = None) (tensor_in)
    tensor_out = L.concatenate([tensor_out, concat_tensor])
    
#    if batch_norm:
#        tensor_out = L.BatchNormalization(momentum = 0.9, epsilon = 1e-5)(tensor_out)
#    
#    tensor_out = insert_activation(tensor_out, activation)
#    
    return tensor_out
    
def pool_layer(tensor_in, n_filters, pool_size, dropout = None, activation = None, batch_norm = False, kern_size = None):

    conv = conv_layer(tensor_in, n_filters, dropout = 0.1, activation = activation, batch_norm = batch_norm, kern_size = kern_size)
    pool = L.MaxPooling2D(pool_size)(conv)
    
    return conv, pool
    
def _expand_inputs(inp, n, var_name):
    # The last n_filter is for bottleneck layer.You can extend this list for U-net with more than 4 pooling layers.
    
    dict_vars = {'n_depth' : [16, 32, 64, 128, 256],\
                 'dropouts': [0.1, 0.1, 0.2, 0.2, 0.3]}
    if var_name in dict_vars.keys():
        if type(inp) is not list:
            inp = np.asarray(dict_vars[var_name])*inp
            inp = inp[:n]
        else:
            if len(inp) != n:
                raise ValueError("Incorrect length for list: %s"%var_name)
    else:
        if type(inp) is tuple:
            inp = [inp]*n
        elif type(inp) is list:
            if len(inp) != n:
                raise ValueError("Incorrect number of elements specified in %s"%var_name)

    return inp
            
    

def build_Unet_flex(img_shape, n_depth = 1, n_pools = 4, activation = 'lrelu', \
                     batch_norm = True, kern_size = (3,3), kern_size_upconv = (2,2), pool_size = (2,2), dropout_level = 1.0,\
                     loss = 'binary_crossentropy', stdinput = True):
    
    # Check inputs
    if (img_shape[0] % (2**n_pools) != 0) | (img_shape[1] % (2**n_pools) != 0):
        raise ValueError("Image shape must be divisible by 2^n_pools.")
    
#    n_filters = np.asarray([16, 32, 64, 128, 256])*n_depth 
    n_filters = _expand_inputs(n_depth, n_pools+1, 'n_depth')
    dropouts = _expand_inputs(dropout_level, n_pools+1, 'dropouts')
    np.asarray([0.1, 0.1, 0.2, 0.2, 0.3])*dropout_level 
    
#    n_filters = n_filters[:n_pools+1]
    dropouts = dropouts[:n_pools+1]
    
    if n_pools not in range(2,5):
        raise ValueError("This implementation allows number of pooling layers to be 2, 3 or 4.")
    
    pool_size = _expand_inputs(pool_size, n_pools, "pool_size")
    kern_size = _expand_inputs(kern_size, n_pools + 1, "kern_size")
    kern_size_upconv = _expand_inputs(kern_size_upconv, n_pools, "kern_size_upconv")

    
    inp = L.Input(img_shape)
    
    if stdinput:
        standardizer = L.Lambda(losses.standardize)
        stdinp = standardizer(inp)
    else:
        stdinp = inp
    
    convs, pools = [], []
    
    for ii in range(n_pools):
        
        if ii == 0:
            tensor_in = stdinp
        else:
            tensor_in = pools[ii-1]
        
        conv, pool = pool_layer(tensor_in, n_filters[ii], pool_size[ii], dropout = dropouts[ii], activation = activation, batch_norm = batch_norm, kern_size = kern_size[ii])
        convs.append(conv)
        pools.append(pool)
    
    bottleneck = conv_layer(pools[-1], n_filters[-1], dropout = dropouts[-1], activation = activation, batch_norm = batch_norm, kern_size = kern_size[ii])
    
    
    for ii in range(n_pools-1, -1, -1):
        
        if ii == n_pools - 1:
            tensor = bottleneck
        
        tensor = upconv_layer(tensor, convs[ii], activation = activation, batch_norm = batch_norm, kern_size = kern_size_upconv[ii], strides = pool_size[ii])
        tensor = conv_layer(tensor, n_filters[ii], dropout = dropouts[ii], activation = activation, batch_norm = batch_norm, kern_size = kern_size[ii])
        
    
    
    out = L.Conv2D(1, (1,1), activation =  'sigmoid') (tensor)
    
    model = keras.models.Model(inputs = inp, outputs = out)
    adam = keras.optimizers.Adam(lr=0.0001, beta_1=0.99, beta_2=0.999, epsilon=1e-8, decay=0.0, amsgrad=False)
    #model.compile(optimizer=adam, loss='binary_crossentropy') # Define the optimizer as adamax GD and loss function as mean squared error
    
    if loss == 'weighted_crossentropy':
        loss = losses.weighted_crossentropy
    elif loss == 'my_binary_crossentropy':
        loss = losses.my_binary_crossentropy
    elif loss == 'focal_loss':
        loss = losses.focal_loss
    elif loss == 'binary_crossentropy':
        loss = loss
    else:
        raise ValueError("Loss function not recognized...")
    
    model.compile(optimizer=adam, loss=loss, metrics = [losses.IoU, losses.acc_zeros, losses.acc_ones])
    
    return model




































#def build_Unet_flex(img_shape, n_depth = 1, n_pools = 4, activation = 'lrelu', \
#                     batch_norm = True, kern_size = (3,3), kern_size_upconv = (2,2), pool_size = (2,2), dropout_level = 1.0,\
#                     loss = 'binary_crossentropy', stdinput = False):
#    
#    n_filters = np.asarray([16, 32, 64, 128, 256])*n_depth
#    dropouts = np.asarray([0.1, 0.1, 0.2, 0.2, 0.3])*dropout_level
#    
#    n_filters = n_filters[:n_pools+1]
#    dropouts = dropouts[:n_pools+1]
#    
#    if n_pools not in range(2,5):
#        raise ValueError("This implementation allows number of pooling layers to be 2, 3 or 4.")
#    
#    inp = L.Input(img_shape)
#    
#    if stdinput:
#        standardizer = L.Lambda(losses.standardize)
#        stdinp = standardizer(inp)
#    else:
#        stdinp = inp
#    
#    convs, pools = [], []
#    
#    for ii in range(n_pools):
#        
#        if ii == 0:
#            tensor_in = stdinp
#        else:
#            tensor_in = pools[ii-1]
#        
#        conv, pool = pool_layer(tensor_in, n_filters[ii], pool_size, dropout = dropouts[ii], activation = activation, batch_norm = batch_norm, kern_size = kern_size)
#        convs.append(conv)
#        pools.append(pool)
#    
#    bottleneck = conv_layer(pools[-1], n_filters[-1], dropout = dropouts[-1], activation = activation, batch_norm = batch_norm, kern_size = kern_size)
#    
#    
#    for ii in range(n_pools-1, -1, -1):
#        
#        if ii == n_pools - 1:
#            tensor = bottleneck
#        
#        tensor = upconv_layer(tensor, convs[ii], activation = activation, batch_norm = batch_norm, kern_size = kern_size_upconv, strides = pool_size)
#        tensor = conv_layer(tensor, n_filters[ii], dropout = dropouts[ii], activation = activation, batch_norm = batch_norm, kern_size = kern_size)
#        
#    
#    
#    out = L.Conv2D(1, (1,1), activation =  'sigmoid') (tensor)
#    
#    model = keras.models.Model(inputs = inp, outputs = out)
#    adam = keras.optimizers.Adam(lr=0.0001, beta_1=0.99, beta_2=0.999, epsilon=1e-8, decay=0.0, amsgrad=False)
#    #model.compile(optimizer=adam, loss='binary_crossentropy') # Define the optimizer as adamax GD and loss function as mean squared error
#    
#    if loss == 'weighted_crossentropy':
#        loss = losses.weighted_crossentropy
#    elif loss == 'my_binary_crossentropy':
#        loss = losses.my_binary_crossentropy
#    elif loss == 'focal_loss':
#        loss = losses.focal_loss
#    elif loss == 'binary_crossentropy':
#        loss = loss
#    else:
#        raise ValueError("Loss function not recognized...")
#    
#    model.compile(optimizer=adam, loss=loss, metrics = [losses.IoU, losses.acc_zeros, losses.acc_ones])
#    
#    return model
#
#
#
