#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines some custom loss functions and metrics that are used to train and evaluate a U-net-like tf.Keras.model. This require the input to be a tensor.

Note: Some of these metrics are implemented in ct_segnet.stats to receive numpy.array as inputs.  

"""


import tensorflow as tf
from tensorflow.keras import backend as K


# Parameters for weighted cross-entropy and focal loss - alpha is higher than 0.5 to emphasize loss in "ones" or metal pixels.
eps = 1e-12
alpha = 0.75
gamma = 2.0

def IoU(y_true, y_pred):
    """
    :return: intersection over union accuracy
    
    Parameters
    ----------
    
    y_true  : tensor
            Ground truth tensor of shape (batch_size, n_rows, n_cols, n_channels)
    y_pred  : tensor
            Predicted tensor of shape (batch_size, n_rows, n_cols, n_channels)
    
    """
#     this is an old implementation that does not assume ignored pixels
#     y_pred = K.round(y_pred)
#     intersection = K.sum(y_true*y_pred)
#     union = K.sum(y_pred) + K.sum(y_true) - intersection
#     acc = (intersection + 1.) / (union + 1.)

#     this implementation assumes ignored pixel labels > 1 in y_true
    y_pred = K.round(y_pred)
    y_pred = tf.cast(y_pred, tf.int32)
    y_true = tf.cast(y_true, tf.int32)
    intersection = K.sum(tf.where(tf.equal(y_true,1), y_true*y_pred, 0))
    union = K.sum(y_pred) + K.sum(tf.where(tf.equal(y_true,1), 1, 0)) - intersection
    intersection = tf.cast(intersection, tf.float32)
    union = tf.cast(union, tf.float32)
    acc = (intersection + 1.) / (union + 1.)
    
    return acc



def acc_zeros(y_true, y_pred):
    """
    :return: accuracy in predicting zero values = TN/(TN + FP)
    
    Parameters
    ----------
    
    y_true  : tensor
            Ground truth tensor of shape (batch_size, n_rows, n_cols, n_channels)
    y_pred  : tensor
            Predicted tensor of shape (batch_size, n_rows, n_cols, n_channels)
    
    """
    
    # Define accuracy of zeros
    y_pred = K.round(y_pred)
    y_pred = tf.cast(y_pred, tf.int32)
    y_true = tf.cast(y_true, tf.int32)

    
#     true_negatives = K.sum((1-y_true)*(1-y_pred))
#     false_positives = K.sum((1-y_true)*y_pred)
#     p = (true_negatives + eps) / (true_negatives + false_positives + eps)

#     This version assumes ignored pixels
    true_negatives = K.sum(tf.where(tf.equal(y_true,0), 1, 0) * (1-y_pred))
    false_positives = K.sum(tf.where(tf.equal(y_true,0), 1 ,0) * y_pred)
    true_negatives = tf.cast(true_negatives, tf.float32)
    false_positives = tf.cast(false_positives, tf.float32)
    p = (true_negatives + eps) / (true_negatives + false_positives + eps)
    
    return p

def acc_ones(y_true, y_pred):
    """
    :return: accuracy in predicting ones = TP/(TP + FN)
    
    Parameters
    ----------
    
    y_true  : tensor
            Ground truth tensor of shape (batch_size, n_rows, n_cols, n_channels)
    y_pred  : tensor
            Predicted tensor of shape (batch_size, n_rows, n_cols, n_channels)
    
    """
    
    # Define accuracy of ones
    y_pred = K.round(y_pred)
    y_pred = tf.cast(y_pred, tf.int32)
    y_true = tf.cast(y_true, tf.int32)

    
    true_positives = K.sum(tf.where(tf.equal(y_true,1), 1, 0)*y_pred)
    false_negatives = K.sum(    tf.where(tf.equal(y_true,1), 1, 0)       * (1-y_pred))
    true_positives = tf.cast(true_positives, tf.float32)
    false_negatives = tf.cast(false_negatives, tf.float32)
    r = (true_positives + eps) / (true_positives + false_negatives + eps)
    
    return r


def _binary_lossmap(y_true, y_pred):
    # y_true, y_pred are tensors of shape (batch_size, img_h, img_w, n_channels)
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return pt_1, pt_0

def my_binary_crossentropy(y_true, y_pred):
    """
    :return: loss value
    
    This is my own implementation of binary cross-entropy. Nothing special.
    
    Parameters
    ----------
    
    y_true  : tensor
            Ground truth tensor of shape (batch_size, n_rows, n_cols, n_channels)
    y_pred  : tensor
            Predicted tensor of shape (batch_size, n_rows, n_cols, n_channels)
    
    """

    pt_1, pt_0 = _binary_lossmap(y_true, y_pred)
    loss_map = -K.log(pt_1 + eps)-K.log(1. - pt_0 + eps)
    return tf.reduce_mean(loss_map)

def focal_loss(y_true, y_pred):
    """
    :return: loss value
    
    Focal loss is defined here: https://arxiv.org/abs/1708.02002
    Using this provides improved fidelity in unbalanced datasets: 
    Tekawade et al. https://doi.org/10.1117/12.2540442
    
    
    Parameters
    ----------
    
    y_true  : tensor
            Ground truth tensor of shape (batch_size, n_rows, n_cols, n_channels)
    y_pred  : tensor
            Predicted tensor of shape (batch_size, n_rows, n_cols, n_channels)
    
    """

    pt_1, pt_0 = _binary_lossmap(y_true, y_pred)
    loss_map = -alpha*K.log(pt_1 + eps)*K.pow(1. - pt_1,gamma) - (1-alpha)*K.log(1. - pt_0 + eps)*K.pow(pt_0,gamma)
    return tf.reduce_mean(loss_map)


def weighted_crossentropy(y_true, y_pred):
    """
    :return: loss value
    
    Weighted cross-entropy allows prioritizing accuracy in a certain class (either 1s or 0s).
    
    Parameters
    ----------
    
    y_true  : tensor
            Ground truth tensor of shape (batch_size, n_rows, n_cols, n_channels)
    y_pred  : tensor
            Predicted tensor of shape (batch_size, n_rows, n_cols, n_channels)
    
    """

    pt_1, pt_0 = _binary_lossmap(y_true, y_pred)
    loss_map = -alpha*K.log(pt_1 + eps)-(1-alpha)*K.log(1. - pt_0 + eps)
    loss_map *= 2.0
    return tf.reduce_mean(loss_map)


def stdize_img(img):
    
    eps = tf.constant(1e-12, dtype = 'float32')
    #mean = tf.reduce_mean(img)
    #img = img - mean
    
    max_ = tf.reduce_max(img)
    min_ = tf.reduce_min(img)

    img = (img - min_ )  / (max_ - min_ + eps)
    return img

def standardize(imgs):
    
    return tf.map_fn(stdize_img, imgs)
    
objects = [IoU, acc_zeros, acc_ones, focal_loss, my_binary_crossentropy, weighted_crossentropy, stdize_img, standardize]

custom_objects_dict = {'tf': tf}
for item in objects:
    custom_objects_dict[item.__name__] = item
