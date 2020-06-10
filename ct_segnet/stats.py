#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A module for estimating signal-to-noise ratio (SNR) for binarizable datasets.

:param numpy.array img: raw input image  

:param numpy.array seg_img: segmented image  

:param numpy.array true_img: ground truth  
"""

import numpy as np
from multiprocessing import cpu_count
import matplotlib as mpl
from ct_segnet.data_utils.data_io import Parallelize

import matplotlib.pyplot as plt

from tomopy import circ_mask

def _get_metalair(img, seg_img):

    """
    input a mask with ones in metal and zeros in air
    """
    img = np.copy(img).astype(np.float32)
    seg_img = seg_img.astype(bool)
    mask = circ_mask(np.ones_like(img, dtype = np.uint8)[np.newaxis,...], 0, ratio = 0.92, val = 0)[0].astype(bool)
    
    metal = seg_img & mask
    air = (~seg_img) & mask
    return metal, air
    
def renormalize(img, seg_img, metal = None, air = None):
    """
    :return: normalized image such that the mean value of segment 1 ("metal") is ~ 1 while that of 0 ("air") is ~ 0.  
    
    """
    if (metal is None) | (air is None):
        metal, air = _get_metalair(img, seg_img)
    min_val = np.mean(img[np.where(air)])
    max_val = np.mean(img[np.where(metal)])
    normed = (img - min_val) / (max_val - min_val)
    return normed

def SNR_data(img, seg_img):
    """
    :return: data vector that calculates SNR (for debugging):
    data = [mean_air, mean_metal, std_air, std_metal]
    SNR = sqrt(mean_metal)/sqrt(std_air^^2 + std_metal^^2)
    """
    metal, air = _get_metalair(img, seg_img)
    img = renormalize(img, seg_img, metal = metal, air = air)
    
    air_img = np.copy(img)
    air_img[np.where(~air)] = np.nan
    
    metal_img = np.copy(img)
    metal_img[np.where(~metal)] = np.nan
    data = [np.nanmean(air_img), np.nanmean(metal_img), np.nanstd(air_img), np.nanstd(metal_img)]
    
    return data

def calc_SNR(img, seg_img):
    """
    :return: float SNR of img w.r.t seg_img
    SNR = sqrt(mean_metal)/sqrt(std_air^^2 + std_metal^^2)
    seg_img is used to estimate mean / std of the segmented phases, called "metal" (pixel value = 1) and "air" (pixel value = 0)
    """
    data = SNR_data(img, seg_img)
    S = np.sqrt((data[1])**2)
    N = np.sqrt((data[2]**2 + data[3]**2)/2.0)
    SNR = S/N
    return SNR



def ROC(thresh, true_img = None, seg_img = None):
    """
    Calculate receiver Operating Characteristics (ROC) curve
    
    Parameters
    ----------
    thresh : float
            threshold value
    true_img : ndarray
            ground truth segmentation map (ny, nx)
    seg_img : ndarray
            predicted segmentation map (ny, nx)
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
    :return: Jaccard accuracy or Intersection over Union  
    
    :rtype: float  
    
    """
    seg_img = np.round(np.copy(seg_img))
    jac_acc = (np.sum(seg_img*true_img) + 1) / (np.sum(seg_img) + np.sum(true_img) - np.sum(seg_img*true_img) + 1)
    return jac_acc

def calc_dice_coeff(true_img, seg_img):
    """
    :return: Dice coefficient  
    
    :rtype: float  
    
    """
    seg_img = np.round(np.copy(seg_img))
    
    dice = (2*np.sum(seg_img*true_img) + 1) / (np.sum(seg_img) + np.sum(true_img) + 1)
    return dice

def fidelity(true_img, seg_img, tolerance = 0.95):
    """
    :return: fidelity  
    
    :rtype: float
    
    Fidelity is number of images with IoU > tolerance
    
    Parameters
    ----------
    tolerance : float
                tolerance (default  = 0.95)
    true_img    : ndarray
                ground truth segmentation map (ny, nx)
    seg_img    : ndarray
                predicted segmentation map (ny, nx)
    
    """

    XY = [(true_img[ii], seg_img[ii]) for ii in range(true_img.shape[0])]
    del true_img
    del seg_img
    
    jac_acc = np.asarray(Parallelize(XY, calc_jac_acc, procs = cpu_count()))
    
    mean_IoU = np.mean(jac_acc)

    jac_fid = np.zeros_like(jac_acc)
    jac_fid[jac_acc > tolerance] = 1
    jac_fid = np.sum(jac_fid).astype(np.float32) / np.size(jac_acc)
    
    
    return jac_fid, mean_IoU, jac_acc






