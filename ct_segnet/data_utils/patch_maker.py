#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 13:34:59 2019

@author: atekawade
"""

import numpy as np







def get_patches(img, patch_size = None, steps = None):
    
    stepy, stepx = steps
    my, mx = img.shape
    py, px = patch_size[0], patch_size[1]
    nx, ny = int(np.ceil(mx/px)), int(np.ceil(my/py))
    
    img = np.asarray([img[ii*stepy:ii*stepy+py] for ii in range(ny)])
    img = np.asarray([[img[jj,:,ii*stepx:ii*stepx+px] for ii in range(nx)] for jj in range(img.shape[0])])
    
    return img


def get_stepsize(img_shape, patch_size):
    
    # Find optimum number of patches to cover full image
    my, mx = img_shape
    py, px = patch_size
    nx, ny = int(np.ceil(mx/px)), int(np.ceil(my/py))
    stepx = (mx-px) // (nx-1) if mx != px else 0
    stepy = (my-py) // (ny-1) if my != py else 0
    
    return (stepy, stepx)



def recon_patches(img, img_shape = None, steps = None):
    
    if img.ndim != 4:
        raise ValueError("Input must be 4D array.")
    
    ny, nx, py, px = img.shape
    stepy, stepx = steps
    
    
    new_img = np.zeros((img_shape))
    
    for ii in range(ny):
        for jj in range(nx):
            new_img[ii*stepy:ii*stepy+py,jj*stepx:jj*stepx+px] = img[ii,jj]
        
    return new_img


def calc_resdrop(img_shape, patch_size, n_max = 3):
    
    y_orig, x_orig, = img_shape
    
    yres = 1
    y_new = y_orig
    while y_new > patch_size[0]*n_max:
        yres += 1
        y_new =  int(np.ceil(y_orig/yres))
    
    xres = 1
    x_new = x_orig
    while x_new > patch_size[1]*n_max:
        xres += 1
        x_new =  int(np.ceil(x_orig/xres))
    
    return yres, xres #, y_new, x_new










def ssrecon_patches(img, img_shape = None, steps = None):
    
    ny, nx = img.shape[:2]
    stepy, stepx = steps
    p = img.shape[-1]
    
    new_img = np.zeros((img_shape))
    
    for ii in range(ny):
        for jj in range(nx):
            new_img[ii*stepy:ii*stepy+p,jj*stepx:jj*stepx+p] = img[ii,jj]
        
    return new_img












