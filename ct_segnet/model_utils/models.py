#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Easily define U-net-like architectures using Keras layers

"""

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers as L
from . import losses

def insert_activation(tensor_in, activation):
    """
    :return: tensor of rank 4 (batch_size, n_rows, n_cols, n_channels)
    
    Parameters
    ----------
    tensor_in : tensor
            input tensor
    activation : str or tf.Keras.layers.Activation
            name of custom activation or Keras activation layer
            
    """
    if activation == 'lrelu':
        tensor_out = L.LeakyReLU(alpha = 0.2)(tensor_in)
    else:
        tensor_out = L.Activation(activation)(tensor_in)
    
    return tensor_out
    


def conv_layer(tensor_in, n_filters, kern_size = None, activation = None, kern_init = 'he_normal', padding = 'same', dropout = 0.1, batch_norm = False):

    """
    Define a block of two convolutional layers
    
    :return: tensor of rank 4 (batch_size, n_rows, n_cols, n_channels)
    
    Parameters
    ----------
    tensor_in  : tensor
            input tensor
    n_filters  : int
            number of filters in each convolutional layer
    kern_size  : tuple
            kernel size, e.g. (3,3)
    activation : str or tf.Keras.layers.Activation
            name of custom activation or Keras activation layer
    kern_init  : str
            kernel initialization method
    padding    : str
            type of padding
    dropout    : float
            dropout fraction
    batch_norm : bool
            True to insert a BN layer
            
    """
    
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
    """
    Define an upconvolutional layer and concatenate the output with a conv layer from the contracting path
    
    :return: tensor of rank 4 (batch_size, n_rows, n_cols, n_channels)
    
    Parameters
    ----------
    tensor_in     : tensor
            input tensor
    concat_tensor : tensor
            this will be concatenated to the output of the upconvolutional layer
    n_filters  : int
            number of filters in each convolutional layer
    kern_size  : tuple
            kernel size for upconv, e.g. (2,2)
    activation : str or tf.Keras.layers.Activation
            name of custom activation or Keras activation layer
    kern_init  : str
            kernel initialization method
    strides    : tuple
            strides e.g. (2,2)
    padding    : str
            type of padding
    batch_norm : bool
            True to insert a BN layer
    
    """
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
    """
    Define a block of 2 convolutional layer followed by a pooling layer
    
    :return: tensor of rank 4 (batch_size, n_rows, n_cols, n_channels)
    
    Parameters
    ----------
    tensor_in     : tensor
            input tensor
    n_filters  : int
            number of filters in each convolutional layer
    pool_size  : tuple
            max pooling (2,2)
    dropout    : float
            fraction of dropout
    activation : str or tf.Keras.layers.Activation
            name of custom activation or Keras activation layer
    batch_norm : bool
            True to insert a BN layer
    kern_size  : tuple
            kernel size for conv layer, e.g. (3,3)
    
    """
    conv = conv_layer(tensor_in, n_filters, dropout = 0.1, activation = activation, batch_norm = batch_norm, kern_size = kern_size)
    pool = L.MaxPooling2D(pool_size)(conv)
    
    return conv, pool
    
def _expand_inputs(inp, n, var_name):
    """
    Handle inputs provided.  
    
    For n_depth and dropouts, if input is a value, return list with length equal to number of pooling layers.  
    
    For kernel sizes, if input is tuple, make a list of the same tuple, for all pooling layers. If it is a list, ensure the length is same as number of pooling layers.  
    
    """
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
    """
    Define your own Unet-like architecture, based on the arguments provided. Checks that the architecture complies with the converging-diverging paths and ensures the output image size is the same as input image size.  
    
    :return: a keras model for a U-net-like architecture
    :rtype: tf.Keras.model
    
    Parameters
    ----------
    img_shape  : tuple
            input image shape (ny,nx,1)
    n_depth  : int or list
            Option 1: a list of the number of filters in each convolutional layer upstream of each pooling layer. Length must equal number of max pooling layers.  
            
            Option 2: an integer that multiplies the values in this list: [16, 32, ...]. E.g. n_depth = 2 creates [32, 64, ...]  
            
    n_pools  : int
            Number of max pooling layers
    activation : str or tf.Keras.layers.Activation
            name of custom activation or Keras activation layer
    batch_norm : bool
            True to insert BN layer after the convolutional layers
    kern_size  : list or tuple
            kernel size, e.g. (3,3). Provide a list (length = n_pools) of tuples to apply a different kernel size to each block.
    kern_size_upconv  : list or tuple
            kernel size for upconv, e.g. (2,2). Provide a list (length = n_pools) of tuples to apply a different kernel size to each block
    pool_size        : list or tuple
            max pool size, e.g. (2,2). Provide a list (length = n_pools) of tuples to apply a different size to each block
    dropout_level : float or list  
            Option 1:  a list (length = n_pools) of dropout values to apply separately in each block  
            
            Option 2:  a float (0..1) that multiples the values in this list: [0.1, 0.1, 0.2, 0.2, 0.3]  
            
    loss          : str
            The loss function of your choice. The following are implemented:  
            'weighted_crossentropy', 'focal_loss', 'binary_crossentropy'  
    stdinput     : bool
            If True, the input image will be normalized into [0,1]
    """

    
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
