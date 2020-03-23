#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 17:42:56 2019

@author: atekawade
"""

import numpy as np
from ImageStackPy import ImageProcessing as IP
from multiprocessing import cpu_count
import matplotlib as mpl

import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture as GMM



def run_augmenter(x_img, y_img, to_do = ["rotate", "flip", "range shift"]):
    
    """
    if aug_type == "rotate":
        f = random_rotate
    elif aug_type == "range shift":
        f = random_range
    elif aug_type == "flip":
        f = random_flip
    """
    f = randomize
    
    XY = [(x_img[ii], y_img[ii]) for ii in range(x_img.shape[0])]
    del x_img
    del y_img
    
    XY = np.asarray(IP.Parallelize(XY, f, to_do = to_do, procs = cpu_count()))
    
    x_img = XY[:,0,:,:]
    y_img = XY[:,1,:,:]
    
    
    return x_img, y_img

def random_range(x_img, y_img):
    
    amin, amax = -1, -1
    while ((amin > 1.0) or (amin < 0.0)): 
        amin = np.random.normal(0.5, 0.25, 1)[0]
    
    while ((amax > 1.0) or (amax < 0.0) or np.abs(amax - amin) < 0.25):
        amax = np.random.normal(0.5, 0.25, 1)[0]
        
    x_img = IP.normalize(x_img, amin = amin, amax = amax)[0]
    
    if amax > amin:
        amax, amin = 1, 0
    else:
        amax, amin = 0, 1
    y_img = IP.normalize(y_img, amin = amin, amax = amax)[0]
        
    return x_img, y_img 



def intensity_flip(x_img, y_img):
    
    if np.random.randint(2,size = 1)[0]:
    
        amin, amax = np.min(x_img), np.max(x_img)
        x_img = IP.normalize(x_img, amin = amax, amax = amin)[0]
        new_y_img = np.zeros_like(y_img)
        new_y_img[y_img == 0] = 1
        return x_img, new_y_img
    else:
        return x_img, y_img
        

def random_flip(x_img, y_img):
    
    flags = np.random.randint(2,size=2)
    
    if flags[0] == 1:
        x_img, y_img = np.flip(x_img, axis = 0), np.flip(y_img, axis = 0)
    if flags[1] == 1:
        x_img, y_img = np.flip(x_img, axis = 1), np.flip(y_img, axis = 1)
        
    return x_img, y_img

        
def random_rotate(x_img, y_img):
    
    nrot = np.random.randint(4, size =1)[0]
    
    return np.rot90(x_img, k = nrot), np.rot90(y_img, k = nrot)
    
    




def randomize(x_img, y_img, to_do = None):

    if "range shift" in to_do:
        x_img, y_img = random_range(x_img, y_img)
    if "rotate" in to_do:
        x_img, y_img = random_rotate(x_img, y_img)
    if "flip" in to_do:
        x_img, y_img = random_flip(x_img, y_img)
    if "intensity flip" in to_do:
        x_img, y_img = intensity_flip(x_img, y_img)
    
    return x_img, y_img

def remove_blanks(x_train, y_train, cutoff = 0.3, return_idx = False):

    
    ystd = np.std(y_train, axis = (1,2))
    fig, ax = plt.subplots(1,1)
    ax.plot(np.sort(ystd))
    ax.plot(np.full(np.shape(ystd), np.max(ystd)*cutoff), label = 'cutoff')
    ax.legend()
    plt.savefig("data_variability_total.png")
    
    if cutoff > 0.0:
        idx = np.where(ystd > np.max(ystd)*cutoff)
        y_train = y_train[idx]
        x_train = x_train[idx]

    if return_idx:
        return x_train, y_train, idx
    else:
        return x_train, y_train
    

def _radial_mask(size, rad, inverted = False):
    
    x = np.arange(-size//2, size//2, 1)
    y = np.copy(x)
    xx, yy = np.meshgrid(x, y)
    zz = np.sqrt(xx**2 + yy**2)
    
    if inverted == False:
        bool_mask = zz < rad
    else:
        bool_mask = zz > rad
    
    mask = np.zeros(bool_mask.shape)
    mask[bool_mask] = 1.0
    
    return mask



def apply_circularmask(img, crop_size = None):
    

    if img.ndim == 3:
        ix, iy = 1,2
    elif img.ndim == 2:
        ix, iy = 0,1
    
    if img.shape[ix] != img.shape[iy]:
            raise ValueError("Image must have a square shape.")
    else:
        size  = img.shape[ix]
        
    if crop_size is None:
        crop_size = int(size*0.05)
    
    mask = _radial_mask(size, size//2 - crop_size)
    
    if img.ndim == 3:
        mask = np.asarray([mask for ii in range(img.shape[0])])
        
    img = img*mask
    
    return img
    

def _add_noise(x_img, degree = 2.0, dist = None):
    
    if dist is None:
        raise ValueError("Required keyword argument dist not provided")
    
    fac = max(0, np.random.normal(degree/2,  degree/2))
    
    gauss = np.random.normal(0, dist*fac, x_img.shape)
    return x_img + gauss


def run_add_noise(X, degree = 2.0, max_points = 1e8):

    n_imgs = int(max_points/(np.prod(X.shape[1:])))
    idx = np.random.choice(X.shape[0], n_imgs)
    data_in = X[idx].copy().reshape(-1,1)    
    print("\tEstimating current noise levels...")
    plt.figure()
    whatever = plt.hist(data_in.reshape(-1), bins = 700)
    plt.savefig('histogram_addnoise.png')
    plt.close()
    model = GMM(2).fit(data_in.reshape(-1,1)) # model size is 2 because binary segmentation
    dist = 0.5*np.abs(model.means_[1][0] - model.means_[0][0])
    print("\tCorrupting images to degree %2.f"%degree)
    X = IP.Parallelize(IP.toStack(X), _add_noise, degree = degree, dist = dist)
    
    return np.asarray(X)
























































    
    
    
    
    