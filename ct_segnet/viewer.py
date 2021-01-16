#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
view segmentation maps overlain with image data
"""

# line 13 empty for good luck

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ct_segnet.data_utils import patch_maker as PM
# from ImageStackPy import ImageProcessing as IP

from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
from skimage.filters import sobel

def view_midplanes(vol = None, ds = None, ax = None, cmap = 'gray', alpha = None, idxs = None, axis_off = False, label_planes = True):
    """View 3 images drawn from planes through axis 0, 1, and 2 at indices listed (idx). Do this for a DataFile or numpy.array
    
    :param matplotlib.axes ax:   three axes  
    
    :param numpy.array vol: 3D numpy array  
    
    :param DataFile ds: If vol is provided, ignore DataFile, else read from this DataFile  
    
    """

    if ax is None:
        fig, ax = plt.subplots(1,3)
    imgs = get_orthoplanes(ds = ds, vol = vol, idxs = idxs)
    for i in range(3):
        ax[i].imshow(imgs[i], cmap = cmap, alpha = alpha)
    
    if label_planes:
        h = ax[0].set_title("XY mid-plane")
        h = ax[1].set_title("XZ mid-plane")
        h = ax[2].set_title("YZ mid-plane")    
    
    if axis_off:
        for ii in range(3):
            ax[ii].axis('off')
    
    return ax

def get_orthoplanes(ds = None, vol = None, idxs = None):
    """Return 3 images drawn from planes through axis 0, 1, and 2 at indices listed (idx). Do this for a DataFile or numpy.array
    
    :return: images at three midplanes  
    
    :rtype: list
    
    :param matplotlib.axes ax:   three axes  
    
    :param numpy.array vol   : 3D numpy array  
    
    :param DataFile ds: If vol is provided, ignore DataFile, else read from this DataFile  
    
    """
    
    if vol is not None:
        if idxs is None: idxs = [vol.shape[i]//2 for i in range(3)]
        imgs = [vol.take(idxs[i], axis = i) for i in range(3)]
    elif ds is not None:
        if idxs is None: idxs = [ds.d_shape[i]//2 for i in range(3)]
        imgs = [ds.read_slice(axis = i, slice_idx = idxs[i]) for i in range(3)]
    
    return imgs    

def edge_plot(img, seg_img, ax, color = [0,255,0]):
    """
    Show edge-map of segmented map overlain on greyscale image.  
    
    :param numpy.array img: grayscale image of any shape (Y,X)  
    
    :param numpy.array seg_img: corresponding segmented image  
    
    :param matplotlib.axes ax: One axis object to plot into  
    
    :param list color: list of the 3 RGB values
    
    """
    
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

    """
    Show edge-map of segmented map overlain on greyscale image.  
    
    :param numpy.array img: grayscale image of any shape (Y,X)  
    
    :param numpy.array seg_img: corresponding segmented image  
    
    :param matplotlib.axes ax: One axis object to plot into  
    
    :param float alpha : blending value
    
    """
    
    
    img = (255*((img - img.min()) / (img.max() - img.min()))).astype(np.uint8)
    ax.imshow(img, cmap = cmap)
    ax.imshow(seg_img, cmap = 'copper', alpha = alpha)
    
    return ax


def add_scalebar(ax, bar_len, resolution, units = 'um', n_dec = 0, pad = 0.35, \
                 fontsize = 16, frameon = True, color = 'black', loc = 'lower center'):
    """
    :param float bar_len: length of bar, e.g. 100 micron  
    
    :param float resolution: pixel size, in units/pixel, e.g. 1.1 microns / pixel  
    
    :param str units: units, e.g. 'um'  
    
    :param int n_dec: number of decimals in number displayed below scalebar  
    
    """

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

