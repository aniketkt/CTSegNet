#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Post-process tomo data after passing through CTSegNet. Do this to remove tessellation artifacts.

This file can be run as a standalone executable or the "morpho_filter" function can be imported into your code.

"""
import time
import sys
import numpy as np


from ct_segnet.data_utils.data_io import DataFile, crops_type, str_to_bool
from ct_segnet.seg_utils import get_repadding

from scipy.ndimage import binary_dilation, binary_erosion
import configargparse


def make_binary_structure(radius, ndim = 3):
    struct_size = 2*radius + 1
    struct = np.ones(tuple([struct_size]*ndim))

    r = np.arange(-radius, radius + 1)

    yy, xx = np.meshgrid(r, r, indexing = 'ij')
    dist = np.sqrt(yy**2 + xx**2)
    struct[dist > radius] = 0
    return struct


@profile
def morpho_filter(vol, ops = ["erode", "dilate"], radius = [5, 5], output = None, crops = None, invert_mask = False):
    """
    this removes tessellation artifacts
    Parameters
    ----------
    vol : np.array
        binary (labeled) volume; can also be a 2D image
    ops : list
        list of operations - pick from "erode" and "dilate"
    radius : radius or list of radii for the respective operations
    invert_mask : bool
        boolean True if mask must be inverted before filters applied
    
    Returns
    -------
    np.array
        filtered binary (labeled) volume
    """

    if invert_mask:
        _message("Inverting mask...")
        vol = vol^1 # invert the labels if required by user. Some models interpret features as voids rather than particles and vice versa.
    
    if crops is not None:
        pads = get_repadding(crops, vol.shape)
        vol = vol[slice(*crops[0]), slice(*crops[1]), slice(*crops[2])]
    
    print("Morphological operations: ")
    if type(radius) is not list:
        radius = [radius]*len(ops)
    
    t00 = time.time()
    for idx, op in enumerate(ops):
        t0 = time.time()
        struct = make_binary_structure(radius[idx], ndim = vol.ndim)
        
        print("%s (r = %i),"%(op, radius[idx]), end = " ")
        if op == "erode":
            vol = binary_erosion(vol, structure = struct, output = output)
        elif op == "dilate":
            vol = binary_dilation(vol, structure = struct, output = output)
        t_step = time.time() - t0
        print("%.1f secs; "%t_step, end = " ")
    
    # Pad the volume to bring it back to original dimensions
    if crops is not None:
        vol = np.pad(vol, pads, 'constant', constant_values = 0)
    
    tot_time = (time.time() - t00)/60.0
    print("\ntotal time for morphological ops = %.2f minutes"%tot_time)
    return vol



def _message(_str):
    print(_str)
    return
    
    
def main(args):
    
    # Create an instance of DataFile of the existing recon
    ds_seg = DataFile(args.input_fname, tiff = False, data_tag = "SEG", VERBOSITY = 0)
    
    
    # Save the output mask to a new file
    ds2 = DataFile(args.output_fname, \
                   data_tag = "SEG", \
                   tiff = False, \
                   d_shape = ds_seg.d_shape, \
                   d_type = ds_seg.d_type, \
                   chunk_shape = ds_seg.chunk_shape,\
                   VERBOSITY = 0)
    ds2.create_new(overwrite = True)
    
    t00 = time.time()
    print("Reading dataset: %s"%ds_seg.fname)
    vol = ds_seg.read_full()
    
    vol = morpho_filter(vol, radius = args.radius, \
                        ops = args.ops, \
                        crop = args.crops, \
                        invert_mask = args.invert_mask)
    
    
    # Save the volume back to the designated DataFile object
    ds2.write_full(vol)
    tot_time_rw = (time.time() - t00)/60.0
    
    print("Done. Total elapsed time (include r/w): %.2f minutes"%tot_time_rw)


if __name__ == "__main__":
    
    
    parser = configargparse.ArgParser()
    
    # list of arguments accepted
    parser.add('-f', '--input_fname', required = True, type = str, help = 'input file name (.hdf5)')
    parser.add('-o', '--output_fname', required = True, type = str, help = 'output file name (.hdf5)')
    parser.add('--ops', required = True, type = str, action = 'append')
    parser.add('--radius', required = True, type = int, action = 'append')
    parser.add('--crops', type = crops_type, action = 'append')
    parser.add('--invert_mask', required = False, type = str_to_bool, default = True, metavar = 'bool', help = 'True to invert mask before applying ops')
    
               
    args = parser.parse_args()
    main(args)
    


    
    
    
    
    
    
    
