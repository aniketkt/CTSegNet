#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 17:42:56 2019

@author: atekawade
"""

import numpy as np
from multiprocessing import cpu_count
import matplotlib as mpl
from ct_segnet.data_utils.data_io import Parallelize

import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture as GMM
from ct_segnet import stats

def _normalize(img, amin = 0.0, amax = 1.0):
    alow = img.min()
    ahigh = img.max()
    
    return np.nan_to_num(amin + (amax-amin)*(img - alow)/(ahigh - alow))

def run_augmenter(x_imgs, y_imgs, to_do = ["rotate", "flip", "gaussian noise", "invert intensity"], nprocs = 1, min_SNR = 2.0, inplace = True):
    
    """
    Parameters
    ----------
    x_imgs : np.array
        input images (n_batch, ny, ny), list of images or 3D numpy array
        
    y_imgs : np.array
        segmented images (n_batch, ny, ny), list of images or 3D numpy array
        
    to_do  : list
        list of strings from "rotate", "flip", "gaussian noise", "invert intensity"
        
    min_SNR : float
        minimum allowable signal-to-noise ratio (SNR)
        
    inplace : bool
        augmentation of data in-place. This is faster and avoids copying the array but could lead to memory-leaks.
        
    """
    if to_do is None: return x_imgs, y_imgs
    
    f = augment_data
    
    if not inplace:
        XY = [(x_imgs[ii], y_imgs[ii]) for ii in range(x_imgs.shape[0])]
        del x_imgs
        del y_imgs

        XY = np.asarray(Parallelize(XY, f, to_do = to_do, min_SNR = min_SNR, procs = nprocs))

        x_imgs = XY[:,0,:,:]
        y_imgs = XY[:,1,:,:]

        return x_imgs, y_imgs
    
    else:
#         print("This loop")
        for i in range(x_imgs.shape[0]):
            x_imgs[i], y_imgs[i] = augment_data(x_imgs[i], y_imgs[i], min_SNR = min_SNR, to_do = to_do)
        return x_imgs, y_imgs

def augment_data(x_img, y_img, to_do = None, min_SNR = 2.0):

    """
    Parameters
    ----------
    x_img : np.array  
        input image (2D)  
        
    y_img : np.array  
        segmentation map (2D)  
    
    Returns
    -------
    tuple
        image pair (input image, segmentation map)
    
    """
    if to_do == None:
        return x_img, y_img
    
    if "gaussian noise" in to_do:
        x_img, y_img = gaussian_noise(x_img, y_img, min_SNR = min_SNR)
    if "rotate" in to_do:
        x_img, y_img = random_rotate(x_img, y_img)
    if "flip" in to_do:
        x_img, y_img = random_flip(x_img, y_img)
    if "intensity flip" in to_do:
        x_img, y_img = invert_intensity(x_img, y_img)
    if "invert intensity" in to_do:
        x_img, y_img = invert_intensity(x_img, y_img)
    
    return x_img, y_img

def invert_intensity(x_img, y_img):
    """
    Darker pixels are made brighter and vice versa. The labels in the corresponding segmentation map are also inverted.
    
    Parameters
    ----------
    x_img : np.array  
        input image (2D)  
        
    y_img : np.array  
        segmentation map (2D)  
    
    Returns
    -------
    tuple
        image pair (input image, segmentation map)
    
    """
    if np.random.randint(2,size = 1)[0]:
        amin, amax = np.min(x_img), np.max(x_img)
        return _normalize(x_img, amin = amax, amax = amin), y_img^1
    else:
        return x_img, y_img

    
    
    
def gaussian_noise(x_img, y_img, min_SNR = 2.0):
    """
    The SNR is first measured in the original image, then normally distributed noise is added such that the SNR does not drop below min_SNR.
    
    Parameters
    ----------
    x_img : np.array  
        input image (2D)  
        
    y_img : np.array  
        segmentation map (2D)  
    
    Returns
    -------
    tuple
        image pair (input image, segmentation map)
    
    """
    SNR0 = stats.calc_SNR(x_img, y_img)
    if min_SNR >= SNR0:
        return x_img, y_img
    else:
        SNR1 = np.random.uniform(min_SNR, SNR0)
        sigma_add = np.sqrt((1/SNR1**2 - 1/SNR0**2))
        return x_img + np.random.normal(0, sigma_add, x_img.shape), y_img

    
def random_flip(x_img, y_img):
    """
    The image and the corresponding seg. map are randomly flipped vertically, horizontally, or both.
    
    Parameters
    ----------
    x_img : np.array  
        input image (2D)  
        
    y_img : np.array  
        segmentation map (2D)  
    
    Returns
    -------
    tuple
        image pair (input image, segmentation map)
    
    """
    
    flags = np.random.randint(2,size=2)
    
    if flags[0] == 1:
        x_img, y_img = np.flip(x_img, axis = 0), np.flip(y_img, axis = 0)
    if flags[1] == 1:
        x_img, y_img = np.flip(x_img, axis = 1), np.flip(y_img, axis = 1)
        
    return x_img, y_img

        
def random_rotate(x_img, y_img):
    """
    The image and the corresponding seg. map are randomly rotated by 90, 180, 270 degrees.
    
    Parameters
    ----------
    x_img : np.array  
        input image (2D)  
        
    y_img : np.array  
        segmentation map (2D)  
    
    Returns
    -------
    tuple
        image pair (input image, segmentation map)
    
    """
    
    nrot = np.random.randint(4, size =1)[0]
    
    return np.rot90(x_img, k = nrot), np.rot90(y_img, k = nrot)
    


def remove_blanks(x_train, y_train, cutoff = 0.3, return_idx = False):

    ystd = np.std(y_train, axis = (1,2))
#     fig, ax = plt.subplots(1,1)
#     ax.plot(np.sort(ystd))
#     ax.plot(np.full(np.shape(ystd), np.max(ystd)*cutoff), label = 'cutoff')
#     ax.legend()
#     plt.savefig("data_variability_total.png")
    
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
#    plt.savefig('histogram_addnoise.png')
    plt.close()
    model = GMM(2).fit(data_in.reshape(-1,1)) # model size is 2 because binary segmentation
    dist = 0.5*np.abs(model.means_[1][0] - model.means_[0][0])
    print("\tCorrupting images to degree %2.f"%degree)
    
    X = Parallelize(X, _add_noise, degree = degree, dist = dist)
    
    return np.asarray(X)


def range_shift(x_img, y_img):
    """
    Not implemented!  
    
    Stretch the contrast range in the image
    
    Parameters
    ----------
    x_img : np.array  
        input image (2D)  
        
    y_img : np.array  
        segmentation map (2D)  
    
    Returns
    -------
    tuple
        image pair (input image, segmentation map)
    
    """
    
#     amin, amax = -1, -1
#     while ((amin > 1.0) or (amin < 0.0)): 
#         amin = np.random.normal(0.5, 0.25, 1)[0]
    
#     while ((amax > 1.0) or (amax < 0.0) or np.abs(amax - amin) < 0.25):
#         amax = np.random.normal(0.5, 0.25, 1)[0]
        
#     x_img = _normalize(x_img, amin = amin, amax = amax)
    
#     if amax > amin:
#         amax, amin = 1, 0
#     else:
#         amax, amin = 0, 1
#     y_img = _normalize(y_img, amin = amin, amax = amax)

    raise ValueError("Not implemented! Consider using gaussian_noise instead.")
    return x_img, y_img 






















































    
    
    
    
    