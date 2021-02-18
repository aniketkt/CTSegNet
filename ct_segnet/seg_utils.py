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
    """
    The Segmenter class wraps over a keras model, integrating 3D slicing and 2D patching functions to enable the 3D-2D-3D conversations in the segmentation workflow.  
    
    model: tf.keras.model  
        keras model with input shape = out shape = (ny, nx, 1)  
    
    model_filename : str  
        path to keras model file (e.g. "model_1.h5")  
    
    model_name : str  
        (optional) just a name for the model  
    
    """
    def __init__(self, model_filename = None, model = None, model_name = "unknown"):
        if model is not None:
            self.model = model
            self.model_name = model_name
        else:
            self.model_name = os.path.split(model_filename)[-1].split('.')[0]
            self.model = load_model(model_filename, custom_objects = custom_objects_dict)

    def seg_image(self, s, max_patches = None, overlap = None):

        """
        Test the segmenter on arbitrary sized 2D image. This method extracts patches of shape same as the input shape of 2D CNN, segments them and stitches them back to form the original image.  
        
        max_patches : tuple  
            (my, mx) are # of patches along Y, X in image  
        
        s : numpy.array  
            greyscale image slice of shape (ny, nx)  
        
        overlap : tuple or int  
            number of overlapping pixels between patches  
        
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
        orig_shape = s.shape
        s = cv2.resize(s, (max_patches[1]*patch_size[1] - overlap[1],\
                           max_patches[0]*patch_size[0] - overlap[0]))
        
        # Make patches
        downres_shape = s.shape
        steps = PM.get_stepsize(downres_shape, patch_size)
        s = PM.get_patches(s, patch_size = patch_size, steps = steps)
        
        # The dataset now has shape: (ny, nx, py, px). ny, nx are # of patches, and py, px is patch_shape.
        # Reshape this dataset into (n, py, px) where n = ny*nx. Trust numpy to preserve order. lol.
        dataset_shape = s.shape
        s = s.reshape((-1,) + patch_size)
        
        # Predict using the model.
        s = self.model.predict(s[...,np.newaxis])
        s = s[...,0]
        
        # Now, reshape the data back...
        s = s.reshape(dataset_shape)
        
        # Reconstruct from patches...
        s = PM.recon_patches(s, img_shape = downres_shape, steps = steps)
        
        # Finally, resize the images to the original shape of slices... This will result in some loss of resolution...
        s = cv2.resize(s, (orig_shape[1], orig_shape[0]))

        # outputs: segmented image of same shape as input image p
        return np.asarray(np.round(s)).astype(np.uint8)  
    

    def seg_chunk(self, p, max_patches = None, overlap = None,\
                  nprocs = None, arr_split = 1, arr_split_infer = 1):
        """
        Segment a volume of shape (nslices, ny, nx). The 2D keras model passes\
        along nslices, segmenting images (ny, nx) with a patch size defined by input \
        to the model  
        
        max_patches: tuple  
            (my, mx) are # of patches along Y, X in image (ny, nx)  

        overlap : tuple or int  
            number of overlapping pixels between patches  
        
        nprocs : int  
            number of CPU processors for multiprocessing Pool  
        
        arr_split : int  
            breakdown chunk into arr_split number of smaller chunks  
        
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
        
        
        p = np.array_split(p, arr_split_infer)
        for jj in range(len(p)):
            p[jj] = self.model.predict(p[jj][...,np.newaxis])[...,0]
            p[jj] = np.round(p[jj])
        p = np.concatenate(p, axis = 0)
        
        p = p.astype(np.uint8) # typecasting
        
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

    """
    
    Returns  
    -------  
        tuple
            padding values to restore 3D np array after it was cropped.  
    
    Parameters  
    ----------  
    
    crops : list
        3 tuples in a list [(nz1,nz2), (ny1,ny2), (nx1,nx2)]  

    d_shape : tuple
        original shape of 3D array  
    
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
    """  
    
    Just a wrapper for cv2's affine transform for rotating an image about center  
    
    Parameters  
    ----------  
    
    imgs : np.array  
        volume or series of images (n, ny, nx)  
    
    angle : float  
        value to rotate image about center, along (ny,nx)  
    
    """
    rows, cols = imgs[0].shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2), angle,1)
    return np.asarray([cv2.warpAffine(imgs[iS],M,(cols,rows)) for iS in range(len(imgs))])    

def process_data(p, segmenter, preprocess_func = None, max_patches = None,\
                 overlap = None, nprocs = None, rot_angle = 0.0, slice_axis = 0,\
                 crops = None, arr_split = 1, arr_split_infer = 1):
    """
    Segment a volume of shape (nz, ny, nx). The 2D keras model passes
    along either axis (0,1,2), segmenting images with a patch size defined by input
    to the model in the segmenter class.  

    Parameters
    ----------
    max_patches : tuple  
        (my, mx) are # of patches along Y, X in image (ny, nx)  

    overlap : tuple or int  
        number of overlapping pixels between patches  

    nprocs : int  
        number of CPU processors for multiprocessing Pool  
    
    arr_split : int  
        breakdown chunk into arr_split number of smaller chunks  
        
    slice_axis : int  
        (0,1,2); axis along which to draw slices  
    

    crops : list  
        list of three tuples; each tuple (start, stop) will define a python slice for the respective axis  
    
    rot_angle : float  
        (degrees) rotate volume around Z axis before slicing along any given axis. Note this is redundant if slice_axis = 0  
        
                      
    nprocs : int  
        number of CPU processors for multiprocessing Pool  
   
    arr_split : int  
        breakdown chunk into arr_split number of smaller chunks  
    
    preprocess_fun : func  
        pass a preprocessing function that applies a 2D filter on an image  
        
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
#         print("\tPreprocessing on XY mapping...")
        p = preprocess_func(p)

    # Run the segmenter algorithm
    p = segmenter.seg_chunk(p, max_patches = max_patches, overlap = overlap, nprocs = nprocs, arr_split = arr_split, arr_split_infer = arr_split_infer)


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



class FeatureExtraction2D(Segmenter):
    '''  
    This class converts a 2D image into an n-dimensional vector z  
    
    Parameters  
    ----------  
    
    max_patches : tuple  
        (my, mx) are # of patches along Y, X in image  


    overlap : tuple or int  
        number of overlapping pixels between patches  
    
    model: tf.keras.model  
        keras model with input shape = out shape = (ny, nx, 1)  
    
    model_filename : str  
        path to keras model file (e.g. "model_1.h5")  
    
    model_name : str  
        (optional) just a name for the model  
    
    '''  
    
    def __init__(self, max_patches = None, overlap = None, model_filename = None):

        
        if type(max_patches) is not tuple:
            max_patches = (max_patches, max_patches)
            
        if type(overlap) is not tuple:
            overlap = (overlap, overlap)
        overlap = (0 if max_patches[0] == 1 else overlap[0],\
                   0 if max_patches[1] == 1 else overlap[1])
        
        self.max_patches = max_patches
        self.overlap = overlap
        self.model_filename = model_filename
#         super(Segmenter, self).__init__(self, model_filename = model_filename)        

        self.model_name = os.path.split(model_filename)[-1].split('.')[0]
        self.model = load_model(model_filename, custom_objects = custom_objects_dict)



    def extract_measurement(self, img, measurement, **kwargs):
        
        '''  
        Returns  
        -------  
        measured_features : np.array  
            shape (ndims,)
        
        Parameters  
        ----------  
        
        img : np.array
            A 2D numpy array (ny,nx). Could be a tomo slice or projection.  
            
        measurement : func  
            function to extract a measurement, e.g. radius, particle centroid, etc.  
            
        '''
        
        if measurement is None:
            raise "ValueError: missing argument measurement is required"
        
        seg_img = self.seg_image(img, max_patches = self.max_patches, overlap = self.overlap)
        
        measured_features = measurement(seg_img, **kwargs)
        return measured_features
        
    def extract_code(s):
        '''
        not implemented  
        
        to do:
        consider patches are created. How should the code vectors of each patch be converted to singe vector? (mean, median, std?)  
        '''  
        raise NotImplementedError
        


    def vis_feature(self, s, measurement, **kwargs):

        """
        This method extracts patches of shape same as the input shape of 2D CNN, measures a feature for each patch's segmentation map and stitches them back to form a checkered image. 
        
        s : numpy.array  
            greyscale image slice of shape (ny, nx)  
        
        """
        
        # Handle patching parameter inputs
        patch_size = self.model.output_shape[1:-1]    
    
        # Resize images
        orig_shape = s.shape
        s = cv2.resize(s, (self.max_patches[1]*patch_size[1] - self.overlap[1],\
                           self.max_patches[0]*patch_size[0] - self.overlap[0]))
        
        # Make patches
        downres_shape = s.shape
        steps = PM.get_stepsize(downres_shape, patch_size)
        s = PM.get_patches(s, patch_size = patch_size, steps = steps)
        
        # The dataset now has shape: (ny, nx, py, px). ny, nx are # of patches, and py, px is patch_shape.
        # Reshape this dataset into (n, py, px) where n = ny*nx. Trust numpy to preserve order.
        dataset_shape = s.shape
        s = s.reshape((-1,) + patch_size)
        
        # Predict using the model.
        s = self.model.predict(s[...,np.newaxis])
        s = s[...,0]
        
        s = np.round(s).astype(np.uint8)
        
        f = [measurement(s[idx], **kwargs) for idx in range(len(s))]
        f = np.asarray(f)
        # the shape of f now is n_imgs, n_features
        nimgs, nfeatures = f.shape
        
        s = np.ones((nimgs, patch_size[0], patch_size[1], nfeatures))
        
        for ife in range(nfeatures):
            s[...,ife] = [s[ii,...,ife]*f[ii,ife] for ii in range(len(s))]
        s = np.asarray(s)
        
        
        
        F_img = []
        for ife in range(nfeatures):
            # Now, reshape the data back...
            f_img = s[...,ife].reshape(dataset_shape)
            # Reconstruct from patches...
            f_img = PM.recon_patches(f_img, img_shape = downres_shape, steps = steps)    
            # Finally, resize the images to the original shape of slices
            f_img = cv2.resize(f_img, (orig_shape[1], orig_shape[0]))            
            F_img.append(f_img)
        
        
        return np.asarray(F_img)
        







if __name__ == "__main__":
    
    message("\n" + "#"*50 + "\n")
    message("Welcome to CTSegNet: AI-based 3D Segmentation\n")
    message("\n" + "#"*50 + "\n")

