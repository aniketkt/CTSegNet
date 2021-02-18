#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example functions for measuring stuff from segmentation masks. Pass these to the FeatureExtraction2D class.  


"""
import numpy as np
import os
import pandas as pd
from ct_segnet import stats


def segmentation_accuracy(seg_img, true_img = None, metrics = ["jaccard", "dice"]):
    
    '''  
    Calculates segmentation accuracy metrics when provided with a ground-truth map.  
    Returns
    -------
    
    np.array  
        vector of accuracy metrics
    
    Parameters
    ----------
    
    seg_img : np.array  
        segmented image (output from Segmenter class)  
        
    true_img : np.array  
        true segmentation map (same shape as seg_img)  
        
    metrics : list  
        list of metric names from "jaccard", "dice", "dice_edge"  
    
    '''
    
    values = []
    for metric in metrics:
        
        if metric == "dice":
            values.append(stats.calc_dice_coeff(true_img, seg_img))
        elif metric == "jaccard":
            values.append(stats.calc_jac_acc(true_img, seg_img))
        
    return np.asarray(values)

from scipy.ndimage import label, find_objects

def pore_analysis(seg_img, features = ["fraction", "number", "size"], invert = False):
    
    '''
    Calculates void fraction, number of pores and mean size of pores. Pore size is simply defined by sqrt of sum of pixels in the pore phase.  
    
    Returns  
    -------  
    np.array  
        vector of measurements  
    
    Parameters   
    ----------  
    seg_img : np.array  
        segmented image (output from Segmenter class)  
        
    true_img : np.array  
        true segmentation map (same shape as seg_img)  
        
    features : list  
        list of porosity features to measure from "fraction", "number" and "size"    
        
    invert : bool  
        if the image contains particles instead of pores, invert = True  
        
    '''
    
    if any(f in features for f in ["number", "size"]):
        img_labeled, n_objs = label(seg_img if invert else seg_img^1)
        obj_list = find_objects(img_labeled)
    
    values = []
    for feature in features:
        if feature == "fraction":
            values.append(1 - np.sum(seg_img)/np.size(seg_img))
        elif feature == "number":
            values.append(n_objs)
        elif feature == "size":
            
            p_size = []
            for idx in range(n_objs):

                sub_img = img_labeled[obj_list[idx]]
                p_area = np.sum(sub_img==(idx+1))                
                p_dia_px = np.sqrt(p_area)#*2*3/(4*np.pi)
                p_size.append(p_dia_px)

            values.append(np.mean(p_size))
        
    return np.asarray(values)

    
