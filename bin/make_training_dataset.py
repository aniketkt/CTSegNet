#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 17:13:22 2019
@author: atekawade
Task: Extract patches of size = model input size from a given CT dataset and manually segmented mask.
Algorithm:
Read Manual Segmentation Mask into memory
for each patch extraction strategy (n_patches) along a given axis (slice_axis):
    draw n_slices along the given axis where n_slices = total_slices / skip_fac
    extract patches of size model_size from each slice
    save data

Repeat this procedure for the grayscale CT Volume
The final .hdf5 file contains two datasets "Y" (segmentation mask) and "X" (images)
The dataset shapes will be n_images x model_size x model_size
"""

import sys
import os
import numpy as np
import pandas as pd

import h5py
import time
import cv2
import gc
import configargparse
from tqdm import tqdm



import matplotlib.pyplot as plt
import matplotlib as mpl
import ast
from ct_segnet.data_utils import data_io
from ct_segnet.data_utils import patch_maker as PM
from ct_segnet.data_utils.data_io import str_to_bool, n_patches_type, crops_type, Parallelize

### HARD INPUTS ####
mpl.use('Agg')

def process_data(p, patch_size = None, n_patches = None, skip_fac = None, axis = None, nprocs = None):
    
    if nprocs is None:
        nprocs = 4
    if p.ndim != 3:
        raise ValueError("Invalid dimensions for 3D data.")


    if type(patch_size) is not tuple:
        patch_size = (patch_size, patch_size)

    if type(n_patches) is not tuple:
        n_patches = (n_patches, n_patches)

#    orig_shape = p[0].shape
    p = np.copy(np.swapaxes(p, 0, axis))[::skip_fac]
#     print("Input data will be resized...")
#     print("\tCurrent d shape:" + str(np.shape(p)))

    p = np.asarray([cv2.resize(p[ii], (n_patches[1]*patch_size[1], n_patches[0]*patch_size[0])) for ii in range(p.shape[0])])
    
    # Make patches
#     print("Making patches...")
#     print("\tCurrent d shape:" + str(np.shape(p)))
    downres_shape = p[0].shape
    steps = PM.get_stepsize(downres_shape, patch_size)
    p = Parallelize(p, PM.get_patches, procs = nprocs, patch_size = patch_size, steps = steps)
    p = np.asarray(p)
    
    
    # The dataset now has shape: (nslices, ny, nx, py, px). ny, nx are # of patches, and py, px is patch_shape.
    # Reshape this dataset into (n, py, px) where n = nslices*ny*nx. Trust numpy to preserve order. lol.
#    dataset_shape = p.shape
    p = p.reshape((-1,) + patch_size)
#     print("Done...")
#     print("\tCurrent d shape:" + str(np.shape(p)))

    return p

def main(args):

    data_io.show_header()
    print("\nLet's make some training data.\n")
    
    # Understand input data format
    ct_istiff = data_io._istiff(args.ct_fpath, args.ct_data_tag)
    seg_istiff = data_io._istiff(args.seg_path, args.seg_data_tag)

    dfile_recon = data_io.DataFile(args.ct_fpath, data_tag = args.ct_data_tag, tiff = ct_istiff, VERBOSITY = 0)
    dfile_seg = data_io.DataFile(args.seg_path, data_tag = args.seg_data_tag, tiff = seg_istiff, VERBOSITY = 0)

    if args.output_fpath == "":
        args.output_fpath = os.path.split(args.seg_path)[0]
    args.output_fname = args.output_fname.split('.')[0] + ".hdf5"
    output_fname = os.path.join(args.output_fpath, args.output_fname)
    
    print("Grayscale CT data:\n%s\n"%dfile_recon.fname)
    dfile_recon.show_stats()
    print("Manually segmented mask:\n%s\n"%dfile_seg.fname)
    dfile_seg.show_stats()
    
    if np.prod(dfile_seg.d_shape) != np.prod(dfile_recon.d_shape):
        raise ValueError("CT data and segmentation mask must be exactly same shape / size")

    # Decide domain extents
    crop_shape = data_io.get_cropped_shape(args.crops, dfile_seg.d_shape)
    
    # Decide patching / slicing strategy
    df_params = pd.DataFrame({"slice axis" : args.slice_axis, "max patches" : args.n_patches})
    skip_fac = args.skip_fac
    # Estimate the shapes of hdf5 data files to be written
    n_images = 0
    for idx, row in df_params.iterrows():
        _len = np.ceil(crop_shape[row["slice axis"]]/skip_fac)
        _len = _len*np.prod(row["max patches"])
        print("Calculated length of set %i: %i"%(idx+1, _len))
        n_images = _len + n_images
    write_dshape = (int(n_images), args.model_size, args.model_size)
    
    # Create datafile objects for writing stuff, and get file paths for it also
    w_dfile_seg = data_io.DataFile(output_fname, data_tag = 'Y', tiff = False, chunked_slice_size = None, d_shape = write_dshape, d_type = np.uint8)
    w_dfile_recon = data_io.DataFile(output_fname, data_tag = 'X', tiff = False, chunked_slice_size = None, d_shape = write_dshape, d_type = np.float32)
    w_dfile_seg.create_new(overwrite = args.overwrite_OK)
    w_dfile_recon.create_new(overwrite = args.overwrite_OK)
    

    # Work on seg data
    print("Working on the segmentation map...")
    d_seg = dfile_seg.read_full()
    d_seg = d_seg[slice(*args.crops[0]), slice(*args.crops[1]), slice(*args.crops[2])]
    gc.collect()
    slice_start = 0
    pbar = tqdm(total = n_images)
    for idx, row in df_params.iterrows():
        p = process_data(d_seg, skip_fac = skip_fac, nprocs = args.nprocs, patch_size = args.model_size, n_patches = row['max patches'], axis = row['slice axis'])
        slice_end = p.shape[0] + slice_start
        s = slice(slice_start, slice_end)
        w_dfile_seg.write_chunk(p, axis = 0, s = s)
        slice_start = slice_end
        del p
        pbar.update(s.stop - s.start)
        gc.collect()
    pbar.close()

    del d_seg
    gc.collect()
        

    # Work on recon data   
    print("Working on the grayscale CT volume...")
    slice_start = 0
    d_recon = dfile_recon.read_full()
    d_recon = d_recon[slice(*args.crops[0]), slice(*args.crops[1]), slice(*args.crops[2])]
    pbar = tqdm(total = n_images)
    for idx, row in df_params.iterrows():
        p = process_data(d_recon, skip_fac = skip_fac, nprocs = args.nprocs, patch_size = args.model_size, n_patches = row['max patches'], axis = row['slice axis'])
        slice_end = p.shape[0] + slice_start
        s = slice(slice_start, slice_end)
        w_dfile_recon.write_chunk(p, axis = 0, s = s)
        slice_start = slice_end
        del p
        pbar.update(s.stop - s.start)
        gc.collect()
    pbar.close()
    


if __name__ == "__main__":

    parser = configargparse.ArgParser()
    parser.add('-c', '--config-setupseg', required=True, is_config_file=True, help='config file for extracting training data')
    parser.add('--model_size', required = True, type = int, help = 'int; in/out image size e.g. 512 for 512x512 image')
    
    parser.add('--ct_fpath', required = True, type = str, help = 'path to CT dataset')
    parser.add('--ct_data_tag', required = False, default = None, type = str, help = 'dataset name/path if hdf5 file')
    parser.add('--seg_path', required = True, type = str, help = 'path to manually segmented mask')
    parser.add('--seg_data_tag', required = False, default = None, type = str, help = 'dataset name/path if hdf5 file')
    parser.add_argument('-n', "--output_fname", required = True, type = str, help = "Name of output hdf5 file.")
    parser.add_argument('-o', "--output_fpath", required = False, default = "", type = str, help = "Parent folder path for saving output. (optional) If not provided, file will be written to same parent folder as input file.")

    parser.add('--n_patches', type = n_patches_type, action = 'append') #ast.literal_eval
    parser.add('--slice_axis', type = int, action = 'append')
    parser.add('--crops', type = crops_type, action = 'append')    
    parser.add('--skip_fac', type = int, required = False, default = 1, help = "reduce number of slices drawn along each axis by this much (> 1)")
    parser.add('--nprocs', required = False, type = int, default = 4, metavar = 'integer', help = 'use these many processors on each subset of chunk read into memory')
    parser.add_argument('-w', "--overwrite_OK", required = False, action = "store_true", default = False, help = "if output file exists, overwrite")
#   
    args = parser.parse_args()

#     print(parser.format_values())
#     print(args)
    
    main(args)
   

    
