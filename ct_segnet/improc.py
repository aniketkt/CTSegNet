#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Plug-n-play functions to pre-process tomo data before passing through CTSegNet.  

"""
import time
import sys
import numpy as np


def modified_autocontrast(vol, s = 0.01, binning = 2):
    '''
    Implementation as in the book by Burge and Burger  
    
    Parameters
    ----------
    vol : np.array
        volume; can also be a 2D image
    s : float or tuple
        quantile of image data to saturate. E.g. s = 0.01 means saturate the lowest 1% and highest 1% pixels. If tuple, interpreted as slow and shigh
    binning : int  
        binning to reduce volume size. speeds up histogram computation
    
    Returns
    -------
    tuple
        alow, ahigh values to clamp data
    
    '''
    
    data_type  = vol.dtype
    
    if type(s) == tuple and len(s) == 2:
        slow, shigh = s
    else:
        slow = s
        shigh = s

    h, bins = np.histogram(vol.astype(np.float32), bins = 500)
    c = np.cumsum(h)
    c_norm = c/np.max(c)
    
    ibin_low = np.argmin(np.abs(c_norm - slow))
    ibin_high = np.argmin(np.abs(c_norm - 1 + shigh))
    
    alow = bins[ibin_low]
    ahigh = bins[ibin_high]
    
    return alow, ahigh


def contrast_adjust(vol, mode = "auto", s = 0.001, binning = 2):
    """  
    
    Parameters  
    ----------  
    vol : np.array  
        volume; can also be a 2D image  
    s : float or tuple  
        quantile of image data to saturate. E.g. s = 0.01 means saturate the lowest 1% and highest 1% pixels. If tuple, interpreted as slow and shigh  
    binning : int  
        binning to reduce volume size. speeds up histogram computation  
    
    Returns  
    -------  
    np.array  
        clipped grayscale volume  
        
    """

    print("contrast adjustment %s"%mode)
    t00 = time.time()
    
    if mode == "auto":
        alow, ahigh = modified_autocontrast(vol, s = s, binning = binning)
    elif mode == "manual":
        alow, ahigh = s
    elif mode == "none":
        return vol
    
    vol = np.clip(vol, alow, ahigh)
    print("\ntotal time for contrast clipping = %.2f seconds"%(time.time() - t00))
    return vol

def _message(_str):
    print(_str)
    return
    
if __name__ == "__main__":
    
    print("nothing here")