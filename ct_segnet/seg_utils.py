#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 17:13:22 2019

@author: atekawade

The run_segmenter_inmem program loads all data into memory before segmenting it, allowing for more manipulation such as 45 deg rotations, tiff data slicing, etc.
v2: This version now handles tiff and hdf5 data formats for input (CT) and output (seg-masks). It has a command-line API with options for file-based inputs.
"""

# line 13 empty for good luck

import sys
import os
import numpy as np
import pandas as pd
import re
import ast
import h5py
import cv2
import time
import tensorflow as tf
import keras
from keras.models import load_model

from ct_segnet.data_utils import patch_maker as PM
from ct_segnet.model_utils.losses import custom_objects_dict

from ImageStackPy import ImageProcessing as IP
VERBOSE = False


import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
sys.stderr = stderr
import absl.logging
logging.root.removeHandler(absl.logging._absl_handler)
absl.logging._warn_preinit_stderr = False
logger = tf.get_logger()
logger.disabled = True
logger.setLevel(logging.FATAL)
graph = tf.get_default_graph()

def message(_str):
    
    if VERBOSE:
        print(_str)
    return


class Segmenter():
    
    def __init__(self, model_filename):

        self.model_name = os.path.split(model_filename)[-1].split('.')[0]
        self.model_filename = model_filename
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.8
        config.gpu_options.allow_growth = True
        session = tf.Session(config = config)
        keras.backend.clear_session() # Always clear session before defining layers. Clears memory and old dependencies.
        self.model = load_model(model_filename, custom_objects = custom_objects_dict)
        
    def seg_image(self, p, max_patches = None, overlap = None):
    
        # Handle patching parameter inputs
        patch_size = self.model.output_shape[1:-1]    
        if type(max_patches) is not tuple:
            max_patches = (max_patches, max_patches)
            
        if type(overlap) is not tuple:
            overlap = (overlap, overlap)
        overlap = (0 if max_patches[0] == 1 else overlap[0], 0 if max_patches[1] == 1 else overlap[1])
    
        # Resize images
        orig_shape = p.shape
        p = cv2.resize(p, (max_patches[1]*patch_size[1] - overlap[1], max_patches[0]*patch_size[0] - overlap[0]))
        
        # Make patches
        downres_shape = p.shape
        steps = PM.get_stepsize(downres_shape, patch_size)
        p = PM.get_patches(p, patch_size = patch_size, steps = steps)
        
        # The dataset now has shape: (ny, nx, py, px). ny, nx are # of patches, and py, px is patch_shape.
        # Reshape this dataset into (n, py, px) where n = ny*nx. Trust numpy to preserve order. lol.
        dataset_shape = p.shape
        p = p.reshape((-1,) + patch_size)
        
        # Predict using the model.
        p = self.model.predict(p[...,np.newaxis])
        p = p[...,0]
        
        # Now, reshape the data back...
        p = p.reshape(dataset_shape)
        
        
        # Reconstruct from patches...
        p = PM.recon_patches(p, img_shape = downres_shape, steps = steps)
        
        # Finally, resize the images to the original shape of slices... This will result in some loss of resolution...
        p = cv2.resize(p, (orig_shape[1], orig_shape[0]))
    
        return np.asarray(np.round(p)).astype(np.uint8)

    def seg_chunk(self, p, max_patches = None, overlap = None, nprocs = None, arr_split = 1):
    
        # Handle patching parameter inputs
        patch_size = self.model.output_shape[1:-1]    
        if type(max_patches) is not tuple:
            max_patches = (max_patches, max_patches)
            
        if type(overlap) is not tuple:
            overlap = (overlap, overlap)
        overlap = (0 if max_patches[0] == 1 else overlap[0], 0 if max_patches[1] == 1 else overlap[1])
    
        # Resize images
        orig_shape = p[0].shape
        p = np.asarray([cv2.resize(p[ii], (max_patches[1]*patch_size[1] - overlap[1], max_patches[0]*patch_size[0] - overlap[0])) for ii in range(p.shape[0])])
        
        # Make patches
        message("Making patches...")
        message("\tCurrent d shape:" + str(np.shape(p)))
        downres_shape = p[0].shape
        steps = PM.get_stepsize(downres_shape, patch_size)
        p = IP.Parallelize(p, PM.get_patches, procs = nprocs, patch_size = patch_size, steps = steps)
        p = np.asarray(p)
        
        # The dataset now has shape: (nslices, ny, nx, py, px). ny, nx are # of patches, and py, px is patch_shape.
        # Reshape this dataset into (n, py, px) where n = nslices*ny*nx. Trust numpy to preserve order. lol.
        dataset_shape = p.shape
        p = p.reshape((-1,) + patch_size)
        
        # Predict using the model.
        message("Running predictions using model...")
        message("\tCurrent d shape:" + str(np.shape(p)))
        p = self.model.predict(p[...,np.newaxis])
        p = p[...,0]
        
        # Now, reshape the data back...
        p = p.reshape(dataset_shape)
        p = [p[ii] for ii in range(p.shape[0])]
        
        # Reconstruct from patches...
        message("Reconstructing from patches...")
        message("\tCurrent d shape:" + str(np.shape(p)))
        p = np.array_split(p, arr_split)
        
        p = [np.asarray(IP.Parallelize(p[ii], PM.recon_patches, img_shape = downres_shape, steps = steps, procs = nprocs)) for ii in range(arr_split)]
        p = np.concatenate(p, axis = 0)
        
        # Finally, resize the images to the original shape of slices... This will result in some loss of resolution...
        message("Resizing images to original slice size...")
        message("\tCurrent d shape:" + str(np.shape(p)))
        p = np.asarray([cv2.resize(p[ii], (orig_shape[1], orig_shape[0])) for ii in range(p.shape[0])])
    
        return np.asarray(np.round(p)).astype(np.uint8)

def get_repadding(crops, d_shape):
    
    pads = []
    for idx, crop in enumerate(crops):
        pad = [0,0]
        if (crop[0] is not None):
            if crop[0] >= 0:
                pad[0] = abs(crop[0])
            elif crop[0] < 0:
                pad[0] = d_shape[idx] - abs(crop[0])
        
        if crop[1] is not None:
            if crop[1] >= 0:
                pad[1] = d_shape[idx] - abs(crop[1])
            elif crop[1] < 0:
                pad[1] = abs(crop[1])
        pads.append(tuple(pad))
        
    return tuple(pads)
                
            

def process_data(p, segmenter, preprocess_func = None, max_patches = None, overlap = None, nprocs = None, rot_angle = 0.0, slice_axis = 0, crops = None, arr_split = 1):
    
    if nprocs is None:
        nprocs = 4
    if p.ndim != 3:
        raise ValueError("Invalid dimensions for 3D data.")

    message("Orienting, rotating and padding as requested...")
    # Rotate the volume along axis 0, if requested
    if rot_angle > 0.0:
        p = np.asarray(IP.rotate_CCW_aboutCenter(p, rot_angle))
    
    if crops is not None:
        
        pads = get_repadding(crops, p.shape)
        p = p[slice(*crops[0]), slice(*crops[1]), slice(*crops[2])]
        

    # Orient the volume such that the first axis is the direction in which to slice through...
    p = np.moveaxis(p, slice_axis, 0)
    message("\tDone")

    # Preprocess function
    if preprocess_func is not None:
        print("\tPreprocessing on XY mapping...")
        p = preprocess_func(p)

    # Run the segmenter algorithm
    p = segmenter.seg_chunk(p, max_patches = max_patches, overlap = overlap, nprocs = nprocs, arr_split = arr_split)


    message("Re-orienting, rotating and padding back original size...")
    # Re-orient the volume such that the first axis is the vertical axis...
    p = np.moveaxis(p, 0, slice_axis)
    

    # Pad the volume to bring it back to original dimensions
    if crops is not None:
        p = np.pad(p, pads, 'constant', constant_values = 0)
    
    # Rotate the volume along axis 0, back to its original state
    if rot_angle > 0.0:
        p = np.asarray(IP.rotate_CCW_aboutCenter(p, -rot_angle))
    message("\tDone")
    
    return p.astype(np.uint8)


if __name__ == "__main__":
    
    message("\n" + "#"*50 + "\n")
    message("Welcome to CTSegNet: AI-based 3D Segmentation\n")
    message("\n" + "#"*50 + "\n")

