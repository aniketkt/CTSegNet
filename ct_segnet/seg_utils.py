#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CTSegNet is more than a 2D CNN model - it's a 3D Segmenter that uses 2D CNNs. The set_utils.py defines the Segmenter class that wraps over a keras U-net-like model (defined by models.py), integrating 3D slicing and 2D patching functions to enable the 3D-2D-3D conversations in the segmentation workflow. To slice a 3D volume, manipulations such as 45 deg rotations, orthogonal slicing, patch extraction and stitching are performed.

"""

import sys
import os
# line 13 empty for good luck
import numpy as np
import pandas as pd
import re
import ast
import h5py
import cv2
import time
import tensorflow as tf
from tensorflow.keras.models import load_model

from ct_segnet.data_utils import patch_maker as PM
from ct_segnet.data_utils.data_io import Parallelize
from ct_segnet.model_utils.losses import custom_objects_dict



VERBOSE = False

def message(_str):
    
    if VERBOSE:
        print(_str)
    return


class Segmenter():
    """The Segmenter class wraps over a keras model, integrating 3D slicing and 2D patching functions to enable the 3D-2D-3D conversations in the segmentation workflow.  
    
    :param model:  keras model with input shape = out shape = (ny, nx, 1)  
    
    :type model: tf.keras.model  
    
    :param str model_filename:  path to keras model file (e.g. "model_1.h5")  
    
    :param str model_name:  (optional) just a name for the model  
    
    """
    def __init__(self, model_filename = None, model = None, model_name = "unknown"):
        if model is not None:
            self.model = model
            self.model_name = model_name
        else:
            self.model_name = os.path.split(model_filename)[-1].split('.')[0]
            self.model = load_model(model_filename, custom_objects = custom_objects_dict)

    def seg_image(self, p, max_patches = None, overlap = None):

        """Test the segmenter on arbitrary sized 2D image. This method extracts patches of shape same as the input shape of 2D CNN, segments them and stitches them back to form the original image.  
        
        :param tuple max_patches: (my, mx) are # of patches along Y, X in image  
        
        :param p: greyscale image of shape (ny, nx)  
        
        :type p: numpy.array
        
        :param overlap: number of overlapping pixels between patches  
        
        :type overlap: tuple or int  
        
        """
        # Handle patching parameter inputs
        patch_size = self.model.output_shape[1:-1]    
        if type(max_patches) is not tuple:
            max_patches = (max_patches, max_patches)
            
        if type(overlap) is not tuple:
            overlap = (overlap, overlap)
        overlap = (0 if max_patches[0] == 1 else overlap[0],\
                   0 if max_patches[1] == 1 else overlap[1])
    
        # Resize images
        orig_shape = p.shape
        p = cv2.resize(p, (max_patches[1]*patch_size[1] - overlap[1],\
                           max_patches[0]*patch_size[0] - overlap[0]))
        
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

        # outputs: segmented image of same shape as input image p
        return np.asarray(np.round(p)).astype(np.uint8)
    

    def seg_chunk(self, p, max_patches = None, overlap = None,\
                  nprocs = None, arr_split = 1):
        """Segment a volume of shape (nslices, ny, nx). The 2D keras model passes\
        along nslices, segmenting images (ny, nx) with a patch size defined by input \
        to the model  
        
        :param tuple max_patches: (my, mx) are # of patches along Y, X in image (ny, nx)

        :param overlap: number of overlapping pixels between patches  
        
        :type overlap: tuple or int  

        :param int nprocs: number of CPU processors for multiprocessing Pool  
        
        :param int arr_split: breakdown chunk into arr_split number of smaller chunks  
        
        """
        
        # Handle patching parameter inputs
        patch_size = self.model.output_shape[1:-1]
        if type(max_patches) is not tuple:
            max_patches = (max_patches, max_patches)
            
        if type(overlap) is not tuple:
            overlap = (overlap, overlap)
        overlap = (0 if max_patches[0] == 1 else overlap[0],\
                   0 if max_patches[1] == 1 else overlap[1])
    
        # Resize images
        orig_shape = p[0].shape
        p = np.asarray([cv2.resize(p[ii], (max_patches[1]*patch_size[1] - overlap[1],\
                                           max_patches[0]*patch_size[0] - overlap[0]))\
                        for ii in range(p.shape[0])])
        
        # Make patches
        message("Making patches...")
        message("\tCurrent d shape:" + str(np.shape(p)))
        downres_shape = p[0].shape
        steps = PM.get_stepsize(downres_shape, patch_size)
        p = Parallelize(p, PM.get_patches, procs = nprocs, \
                           patch_size = patch_size, steps = steps)
        p = np.asarray(p)
        
        # The dataset now has shape: (nslices, ny, nx, py, px),
        # where ny, nx are # of patches, and py, px is patch_shape.
        # Reshape this dataset into (n, py, px) where n = nslices*ny*nx.
        dataset_shape = p.shape
        p = p.reshape((-1,) + patch_size)
        
        # Predict using the model.
        message("Running predictions using model...")
        message("\tCurrent d shape:" + str(np.shape(p)))
        p = self.model.predict(p[...,np.newaxis])
        p = p[...,0]
        p = np.asarray(np.round(p)).astype(np.uint8)
        
        # Now, reshape the data back...
        p = p.reshape(dataset_shape)
        p = [p[ii] for ii in range(p.shape[0])]
        
        # Reconstruct from patches...
        message("Reconstructing from patches...")
        message("\tCurrent d shape:" + str(np.shape(p)))
        p = np.array_split(p, arr_split)
        
        p = [np.asarray(Parallelize(p[ii], PM.recon_patches,\
                                    img_shape = downres_shape,\
                                    steps = steps, procs = nprocs\
                                   )) for ii in range(arr_split)]
        p = np.concatenate(p, axis = 0)
        
        # Finally, resize the images to the original shape of slices... This will result in some loss of resolution...
        message("Resizing images to original slice size...")
        message("\tCurrent d shape:" + str(np.shape(p)))
        p = np.asarray([cv2.resize(p[ii], (orig_shape[1], orig_shape[0]))\
                        for ii in range(p.shape[0])])
        return p

def get_repadding(crops, d_shape):

    """Returns padding values to restore 3D np array after it was cropped.  
    
    :param list crops: 3 tuples in a list [(nz1,nz2), (ny1,ny2), (nx1,nx2)]  
    
    :param tuple d_shape: original shape of 3D array  
    
    """
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

                
def _rotate(imgs, angle):
    """Just a wrapper for cv2's affine transform for rotating an image about center  
    
    :param imgs:   volume or series of images (n, ny, nx)  
    
    :type imgs: numpy.array  
    
    :param float angle: value to rotate image about center, along (ny,nx)  
    
    """
    rows, cols = imgs[0].shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2), angle,1)
    return np.asarray([cv2.warpAffine(imgs[iS],M,(cols,rows)) for iS in range(len(imgs))])    
def process_data(p, segmenter, preprocess_func = None, max_patches = None,\
                 overlap = None, nprocs = None, rot_angle = 0.0, slice_axis = 0,\
                 crops = None, arr_split = 1):
    """Segment a volume of shape (nz, ny, nx). The 2D keras model passes
    along either axis (0,1,2), segmenting images with a patch size defined by input
    to the model in the segmenter class.  

    :param tuple max_patches: (my, mx) are # of patches along Y, X in image (ny, nx)

    :param overlap: number of overlapping pixels between patches  

    :type overlap: tuple or int  

    :param int nprocs: number of CPU processors for multiprocessing Pool  

    :param int arr_split: breakdown chunk into arr_split number of smaller chunks  

    
    :param int slice_axis: (0,1,2); axis along which to draw slices  
    
    :param list crops: list of three tuples; each tuple (start, stop) will
                      define a python slice for the respective axis  
                      
    :param float rot_angle: (degrees) rotate volume around Z axis before slicing along any given axis. Note this is redundant if slice_axis = 0  
    
    :param int nprocs: number of CPU processors for multiprocessing Pool
    
    :param int arr_split: breakdown chunk into arr_split number of smaller chunks  
    
    :param func preprocess_func: pass a preprocessing function that applies a 2D filter on an image  
    
    """
    if nprocs is None:
        nprocs = 4
    if p.ndim != 3:
        raise ValueError("Invalid dimensions for 3D data.")

    message("Orienting, rotating and padding as requested...")
    # Rotate the volume along axis 0, if requested
    if rot_angle > 0.0:
        p = _rotate(p, rot_angle)
    
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
        p = _rotate(p, -rot_angle)
    message("\tDone")
    
    return p.astype(np.uint8)


if __name__ == "__main__":
    
    message("\n" + "#"*50 + "\n")
    message("Welcome to CTSegNet: AI-based 3D Segmentation\n")
    message("\n" + "#"*50 + "\n")

