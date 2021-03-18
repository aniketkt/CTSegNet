#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A module for estimating
    1. signal-to-noise ratio (SNR) for binarizable datasets.  
    2. accuracy metrics for segmentation maps.  

"""

import numpy as np
from multiprocessing import cpu_count
import matplotlib as mpl
from ct_segnet.data_utils.data_io import Parallelize

import matplotlib.pyplot as plt


def calc_SNR(img, seg_img, labels = (0,1), mask_ratio = None):
    """
    SNR =  1     /  s*sqrt(std0^^2 + std1^^2)  
    where s = 1 / (mu1 - mu0)  
    mu1, std1 and mu0, std0 are the mean / std values for each of the segmented regions respectively (pix value = 1) and (pix value = 0).  
    seg_img is used as mask to determine stats in each region.  

    Parameters
    ----------
    img : np.array  
        raw input image (2D or 3D)  
    
    seg_img : np.array  
        segmentation map (2D or 3D)  
        
    labels : tuple  
        an ordered list of two label values in the image. The high value is interpreted as the signal and low value is the background.  
        
    mask_ratio : float or None
        If not None, a float in (0,1). The data are cropped such that the voxels / pixels outside the circular mask are ignored.  

    Returns
    -------
    float
        SNR of img w.r.t seg_img  

    """
    
    # handle circular mask  
    if mask_ratio is not None:
        crop_val = int(img.shape[-1]*0.5*(1 - mask_ratio/np.sqrt(2)))
        crop_slice = slice(crop_val, -crop_val)    

        if img.ndim == 2: # 2D image
            img = img[crop_slice, crop_slice]
            seg_img = seg_img[crop_slice, crop_slice]
        elif img.ndim == 3: # 3D image
            vcrop = int(img.shape[0]*(1-mask_ratio))
            vcrop_slice = slice(vcrop, -vcrop)
            img = img[vcrop_slice, crop_slice, crop_slice]
            seg_img = seg_img[vcrop_slice, crop_slice, crop_slice]
            
        
    pix_1 = img[seg_img == labels[1]]
    pix_0 = img[seg_img == labels[0]]
    mu1 = np.mean(pix_1)
    mu0 = np.mean(pix_0)
    s = abs(1/(mu1 - mu0))
    std1 = np.std(pix_1)
    std0 = np.std(pix_0)
    std = np.sqrt(0.5*(std1**2 + std0**2))
    std = s*std
    return 1/std


def ROC(thresh, true_img = None, seg_img = None):
    """
    Receiver Operating Characteristics (ROC) curve  
    
    Parameters
    ----------
    thresh : float
            threshold value
            
    true_img : numpy.array
            ground truth segmentation map (ny, nx)
            
    seg_img : numpy.array
            predicted segmentation map (ny, nx)
    
    Returns
    -------
    tuple
        FPR, TPR        
            
    """
    y_p = np.zeros_like(seg_img)
    y_p[seg_img > thresh] = 1
    true_img = np.copy(true_img)

    TN = np.sum((1-true_img)*(1-y_p)).astype(np.float32)
    
    FP = np.sum((1-true_img)*y_p).astype(np.float32)
    
    TNR = TN / (TN + FP)
    FPR = 1 - TNR
    
    TP = np.sum(true_img*y_p).astype(np.float32)
    
    FN = np.sum(true_img*(1-y_p)).astype(np.float32)
    
    TPR = TP / (TP +  FN)
    
    return (FPR, TPR)

def calc_jac_acc(true_img, seg_img):
    """
    Parameters
    ----------
    true_img : np.array
            ground truth segmentation map (ny, nx)
            
    seg_img : np.array
            predicted segmentation map (ny, nx)
     
    Returns
    -------
    float
        Jaccard accuracy or Intersection over Union  
    """
    seg_img = np.round(np.copy(seg_img))
    
    jac_acc = (np.sum(seg_img*(true_img == 1)) + 1) / (np.sum(seg_img) + np.sum((true_img == 1)) - np.sum(seg_img*(true_img == 1)) + 1)
    return jac_acc

def calc_dice_coeff(true_img, seg_img):
    """
    Parameters
    ----------
    true_img : np.array
            ground truth segmentation map (ny, nx)
            
    seg_img : np.array
            predicted segmentation map (ny, nx)
     
    Returns
    -------
    float
        Dice coefficient  

    """
    seg_img = np.round(np.copy(seg_img))
    
    dice = (2*np.sum(seg_img*(true_img == 1)) + 1) / (np.sum(seg_img) + np.sum((true_img == 1)) + 1)
    return dice

def fidelity(true_imgs, seg_imgs, tolerance = 0.95):
    """
    Fidelity is number of images with IoU > tolerance  
    
    Parameters
    ----------
    tolerance : float
                tolerance (default  = 0.95)
                
    true_imgs : numpy.array
                list of ground truth segmentation maps (nimgs, ny, nx)
                
    seg_imgs  : numpy.array
                list of predicted segmentation maps (nimgs, ny, nx)
     
    Returns
    -------
    float
        Fidelity  
    """

    XY = [(true_imgs[ii], seg_imgs[ii]) for ii in range(true_imgs.shape[0])]
    del true_imgs
    del seg_imgs
    
    jac_acc = np.asarray(Parallelize(XY, calc_jac_acc, procs = cpu_count()))
    
    mean_IoU = np.mean(jac_acc)

    jac_fid = np.zeros_like(jac_acc)
    jac_fid[jac_acc > tolerance] = 1
    jac_fid = np.sum(jac_fid).astype(np.float32) / np.size(jac_acc)
    
    
    return jac_fid, mean_IoU, jac_acc






