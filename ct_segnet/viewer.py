#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 17:13:22 2019

@author: atekawade

This file contains useful functions to view segmentation maps
"""

# line 13 empty for good luck

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ct_segnet.data_utils import patch_maker as PM
from ct_segnet.model_utils.losses import custom_objects_dict
# from ImageStackPy import ImageProcessing as IP

from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
from skimage.filters import sobel

def view_midplanes(vol = None, ds = None, ax = None, cmap = 'gray', alpha = None, idxs = None):
    # ax    :   three axes
    # vol   : 3D numpy array
    # ds    : DataFile object

    if ax is None:
        fig, ax = plt.subplots(1,3)
    imgs = get_orthoplanes(ds = ds, vol = vol, idxs = idxs)
    for i in range(3):
        ax[i].imshow(imgs[i], cmap = cmap, alpha = alpha)
    
    return ax

def get_orthoplanes(ds = None, vol = None, idxs = None):
    
    if vol is not None:
        if idxs is None: idxs = [vol.shape[i]//2 for i in range(3)]
        imgs = [vol.take(idxs[i], axis = i) for i in range(3)]
    elif ds is not None:
        if idxs is None: idxs = [ds.d_shape[i]//2 for i in range(3)]
        imgs = [ds.read_slice(axis = i, slice_idx = idxs[i]) for i in range(3)]
    
    return imgs    

def edge_plot(img, seg_img, ax, color = [0,255,0]):
    # img: grayscale image of any shape (Y,X)
    # seg_img: corresponding segmented image
    # ax: matplotlib axis
    # color: RGB values
    img = (255*((img - img.min()) / (img.max() - img.min()))).astype(np.uint8)
    img = np.concatenate([img[...,np.newaxis]]*3, axis = -1)
#     sob_img = IP.calc_sobel(np.copy(seg_img))[0]
    sob_img = sobel(np.copy(seg_img))
    sob_img = (sob_img - sob_img.min()) / (sob_img.max() - sob_img.min())
    sob_img = (255*np.round(sob_img)).astype(np.uint8)
    img[sob_img == 255] = color

    ax.imshow(img)
    
    return ax

def seg_plot(img, seg_img, ax, alpha = 0.3, cmap = 'gray'):
    # img: grayscale image of any shape (Y,X)
    # seg_img: corresponding segmented image
    # ax: matplotlib axis
    # alpha : blending value
    img = (255*((img - img.min()) / (img.max() - img.min()))).astype(np.uint8)
    ax.imshow(img, cmap = cmap)
    ax.imshow(seg_img, cmap = 'copper', alpha = alpha)
    
    return ax


def add_scalebar(ax, bar_len, resolution, units = 'um', n_dec = 0, pad = 0.35, \
                 fontsize = 16, frameon = True, color = 'black', loc = 'lower center'):
    
    # bar_len      : float, length of bar, e.g. 100 micron
    # resolution   : float, pixel size, in units/pixel, e.g. 1.1 microns / pixel
    # units        : str, what units?
    # n_dec        : number of decimals in number displayed below scalebar

    fontprops = fm.FontProperties(size=fontsize)
    n_pix = bar_len/resolution
    scalebar = AnchoredSizeBar(ax.transData, n_pix, ("%."+ "%if"%n_dec)%bar_len + " " + units, loc, \
                               pad=pad, color=color, frameon=frameon, size_vertical=1, fontproperties=fontprops)
    ax.add_artist(scalebar)
    return ax



if __name__ == "__main__":
    
    print("\n" + "#"*50 + "\n")
    print("Welcome to CTSegNet: AI-based 3D Segmentation\n")
    print("\n" + "#"*50 + "\n")

