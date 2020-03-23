#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 13:34:59 2019

@author: atekawade
"""

import numpy as np
from scipy.ndimage import affine_transform





def resize_volume(vol, new_shape, order = None):

    old_shape = vol.shape
    s = np.asarray([float(old_shape[i])/float(new_shape[i]) for i in range(3)])
    
    M = np.identity(3)*s
    return affine_transform(vol, M, np.zeros(3), order = order, output_shape = new_shape)


def get_patches(vol, patch_size = None, steps = None):
    
    stepz, stepy, stepx = steps
    mz, my, mx = vol.shape
    pz, py, px = patch_size[0], patch_size[1], patch_size[2]
    nx, ny, nz = int(np.ceil(mx/px)), int(np.ceil(my/py)), int(np.ceil(mz/pz))
    
    vol = np.asarray([vol[ii*stepz:ii*stepz+pz] for ii in range(nz)])
    vol = np.asarray([[vol[jj,:,ii*stepy:ii*stepy+py] for ii in range(ny)] for jj in range(vol.shape[0])])
    vol = np.asarray([[[vol[jj,kk,:,:,ii*stepx:ii*stepx+px] for ii in range(nx)] for kk in range(vol.shape[1])] for jj in range(vol.shape[0])])
    return vol


def get_stepsize(vol_shape, patch_size):
    
    # Find optimum number of patches to cover full image
    mz, my, mx = vol_shape
    pz, py, px = patch_size
    nx, ny, nz = int(np.ceil(mx/px)), int(np.ceil(my/py)), int(np.ceil(mz/pz))
    stepx = (mx-px) // (nx-1) if mx != px else 0
    stepy = (my-py) // (ny-1) if my != py else 0
    stepz = (mz-pz) // (nz-1) if mz != pz else 0
    
    return (stepz, stepy, stepx)



def recon_patches(vol, vol_shape = None, steps = None):
    
    if vol.ndim != 6:
        raise ValueError("Input must be 6D array.")
    
    nz, ny, nx, pz, py, px = vol.shape
    stepz, stepy, stepx = steps
    
    
    new_vol = np.zeros((vol_shape), dtype = vol.dtype)
    
    for kk in range(nz):
        for ii in range(ny):
            for jj in range(nx):
                new_vol[kk*stepz:kk*stepz+pz, ii*stepy:ii*stepy+py, jj*stepx:jj*stepx+px] = vol[kk,ii,jj]
        
    return new_vol







