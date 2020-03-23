#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 14:14:34 2019

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

eps = 1e-12
alpha = 0.75
gamma = 2.0

def IoU(y_true, y_pred):
    
    y_pred = K.round(y_pred)
    intersection = K.sum(y_true*y_pred)
    union = K.sum(y_pred) + K.sum(y_true) - intersection
    acc = (intersection + 1.) / (union + 1.)
    return acc



def acc_zeros(y_true, y_pred):
    
    y_pred = K.round(y_pred)
    #true_positives = K.sum(y_true*y_pred)
    #false_positives = K.sum((1-y_true)*y_pred)
    #p = (true_positives + eps) / (true_positives + false_positives + eps)
    
    true_negatives = K.sum((1-y_true)*(1-y_pred))
    false_positives = K.sum((1-y_true)*y_pred)
    p = (true_negatives + eps) / (true_negatives + false_positives + eps)
    return p

def acc_ones(y_true, y_pred):
    
    y_pred = K.round(y_pred)
    true_positives = K.sum(y_true*y_pred)
    false_negatives = K.sum(y_true*(1-y_pred))
    r = (true_positives + eps) / (true_positives + false_negatives + eps)
    return r


#
#def focal_loss(gamma=2., alpha=.25):
#    def loss_func(y_true, y_pred):
#        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
#        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
#        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))
#    return loss_func



def focal_loss(y_true, y_pred):

    pt_1, pt_0 = binary_lossmap(y_true, y_pred)
    loss_map = -alpha*K.log(pt_1 + eps)*K.pow(1. - pt_1,gamma) - (1-alpha)*K.log(1. - pt_0 + eps)*K.pow(pt_0,gamma)
    return tf.reduce_mean(loss_map)


def binary_lossmap(y_true, y_pred):
    
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return pt_1, pt_0

def my_binary_crossentropy(y_true, y_pred):

    pt_1, pt_0 = binary_lossmap(y_true, y_pred)
    loss_map = -K.log(pt_1 + eps)-K.log(1. - pt_0 + eps)
    return tf.reduce_mean(loss_map)


def weighted_crossentropy(y_true, y_pred):

    pt_1, pt_0 = binary_lossmap(y_true, y_pred)
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
    




objects = [IoU, acc_zeros, acc_ones, focal_loss, my_binary_crossentropy, weighted_crossentropy, standardize, stdize_img]

custom_objects_dict = {'tf': tf}
for item in objects:
    custom_objects_dict[item.__name__] = item
