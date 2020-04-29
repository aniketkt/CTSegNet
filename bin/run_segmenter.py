#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 17:13:22 2019

@author: atekawade
Task: Run 3D segmenter on an arbitrarily sized CT dataset. 
Algorithm Description:
run_segmenter.py is optimized for minimal use of disk r/w actions, when datasets are smaller or you have lots of RAM. As a thumb-rule, this works well when dataset size (32 bit float) is < 20% of RAM available.
The full CT volume is read into memory, then from a given "slice_axis", patches are extracted as given by "n_patches", and passed into the fCNN. The segmented volume is then reconstructed from the segmented patches output from fCNN. This generates one mask. This is repeated for as many masks along various combinations of slice_axis, n_patches. As opposed to the hdf5-only version, this version reads the full volume into memory allowing additional features; you can:
1. rotate the volume along axis 0, then slice along any axis 1 or 2.
2. crop out part of the volume and segment it, then restore it into the native index space

The above process makes any number of segmentation maps. Finally, an ensemble vote of all these masks is performed.

Algorithm:
if run_seg is True:
    read CT volume into memory
    for mask in masks:
        create new dataset with name = mask_name
        segment volume with process_data() function
        write volume into tiff / hdf5 named (mask_name)
if run_ensemble is True: # same as run_segmenter_hdf5
    create new hdf5 or tiff folder (tiff_output = False or True)
        for mask in masks:
            read chunk
        calculate voxel-wise median for all chunks from respective masks
        write chunk into new hdf5 or tiff folder (vote_maskname)
    if remove_masks is True:
        delete all masks in list mask_name
    
"""
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
from shutil import rmtree

import numpy as np
import pandas as pd
import configargparse
from tqdm import tqdm

import ast
import h5py
import cv2
import time

from ct_segnet.model_utils.losses import custom_objects_dict
from ct_segnet.data_utils import patch_maker as PM
from ct_segnet.seg_utils import Segmenter, process_data

import matplotlib.pyplot as plt
import matplotlib as mpl

from ct_segnet.data_utils import data_io
from ct_segnet.data_utils.data_io import str_to_bool, n_patches_type, crops_type


def main(args):
    # Build mask parameters DataFrame
    df_params = pd.DataFrame({"mask_name" : args.mask_name,\
                              "slice_axis" : args.slice_axis,\
                              "n_patches" : args.n_patches,\
                              "overlap"  : args.overlap, \
                              "rotation" : args.rotation})
#     print(df_params)
    
    mpl.use(args.mpl_agg)


    data_io.show_header()
    if not os.path.exists(args.seg_path): os.makedirs(args.seg_path)
    if args.run_seg:
        
        # Understand input data format
        if os.path.isdir(args.ct_fpath):
            tiff_input = True
        elif args.ct_fpath.split('.')[-1] in ("hdf5", "h5"):
            tiff_input = False
            if args.ct_data_tag == "":
                raise ArgumentTypeError("dataset-name required for hdf5")
        else:
            raise ArgumentTypeError("input file type not recognized. must be tiff folder or hdf5 file")
        
        ct_dfile = data_io.DataFile(args.ct_fpath, \
                                    tiff = tiff_input,\
                                    data_tag = args.ct_data_tag, \
                                    VERBOSITY = args.rw_verbosity)
        ct_dfile.show_stats()
        chunk_shape = ct_dfile.chunk_shape
        if args.stats_only:
            print("\nSet stats_only = False and start over to run program.")
            sys.exit()
        
        # Load model from model repo
        model_filename = os.path.join(args.model_path, args.model_name + '.hdf5')
        print("\nStarting segmentation mode ...")
        segmenter = Segmenter(model_filename = model_filename)
        
        print("Reading CT volume into memory...")
        dd = ct_dfile.read_full()
        
        for idx, row in df_params.iterrows(): # iterate over masks

            # assign arguments from df_params for this mask
            slice_axis = row["slice_axis"]
            max_patches = row["n_patches"]
            segfile_tag = row["mask_name"]
            overlap = row["overlap"]
            rotation = row["rotation"]
            
            # define DataFile object for mask
            seg_fname = os.path.join(args.seg_path, segfile_tag)
            if not args.tiff_output: seg_fname = seg_fname + ".hdf5"
            seg_dfile = data_io.DataFile(seg_fname, \
                                         data_tag = "SEG",\
                                         tiff = args.tiff_output, \
                                         d_shape = ct_dfile.d_shape, \
                                         d_type = np.uint8, \
                                         chunk_shape = chunk_shape,\
                                         VERBOSITY = args.rw_verbosity)            
            seg_dfile.create_new(overwrite = args.overwrite_OK)

            t0 = time.time()
            
            print("\nWorking on %s\n"%segfile_tag)
            ch = process_data(dd, segmenter, \
                              slice_axis = slice_axis, \
                              rot_angle = rotation, \
                              max_patches = max_patches, \
                              overlap = overlap, \
                              nprocs = args.nprocs, \
                              arr_split = args.arr_split, \
                              crops = args.crops)
            seg_dfile.write_full(ch)
            t1 = time.time()
            total_time = (t1 - t0) / 60.0
            print("\nDONE on %s\nTotal time for generating %s mask: %.2f minutes"%(time.ctime(), segfile_tag, total_time))
            del slice_axis
            del max_patches
            del segfile_tag
            del rotation
            del ch
        
    if args.run_ensemble:
        print("\nStarting ensemble mode ...\n")

        t0 = time.time()
        # get the d_shape of one of the masks
        temp_fname = os.path.join(args.seg_path, df_params.loc[0,"mask_name"])
        if not args.tiff_output: temp_fname = temp_fname + ".hdf5"
        temp_ds = data_io.DataFile(temp_fname, data_tag = "SEG", tiff = args.tiff_output, VERBOSITY = 0)
        mask_shape = temp_ds.d_shape
        chunk_shape = temp_ds.chunk_shape
        if not args.run_seg: temp_ds.show_stats()
        del temp_ds
        del temp_fname

        if args.stats_only:
            print("\nSet stats_only = False and start over to run program.")
            sys.exit()
        
        vote_fname = os.path.join(args.seg_path, args.vote_maskname)
        if not args.tiff_output: vote_fname = vote_fname + ".hdf5"
        vote_dfile = data_io.DataFile(vote_fname, \
                                      tiff = args.tiff_output,\
                                      data_tag = "SEG",\
                                      d_shape = mask_shape, \
                                      d_type = np.uint8, \
                                      chunk_shape = chunk_shape,\
                                      VERBOSITY = args.rw_verbosity)            
        vote_dfile.create_new(overwrite = args.overwrite_OK)
        
        slice_start = 0
        n_masks = len(df_params)
        pbar = tqdm(total = mask_shape[0])
        while slice_start < mask_shape[0]:
            ch = [0]*len(df_params)
            for idx, row in df_params.iterrows():
                seg_fname = os.path.join(args.seg_path, row["mask_name"])
                if not args.tiff_output: seg_fname = seg_fname + ".hdf5"

                
                seg_dfile = data_io.DataFile(seg_fname, \
                                             tiff = args.tiff_output, \
                                             data_tag = "SEG", \
                                             VERBOSITY = args.rw_verbosity)        
                if mask_shape != seg_dfile.d_shape:
                    raise ValueError("Shape of all masks must be same")
                    
                ch[idx], s = seg_dfile.read_chunk(axis = 0, \
                                                  slice_start = slice_start, \
                                                  max_GB = args.mem_thres/(n_masks))
            ch = np.asarray(ch)
            ch = np.median(ch, axis = 0).astype(np.uint8)
            vote_dfile.write_chunk(ch, axis = 0, s = s)
            del ch
            slice_start = s.stop            
            pbar.update(s.stop - s.start)
        pbar.close()

        t1 = time.time()
        total_time = (t1 - t0) / 60.0
        print("\nDONE on %s\nTotal time for ensemble mask %s : %.2f minutes"%(time.ctime(), args.vote_maskname, total_time))


    if args.remove_masks:
        print("Intermediate masks will be removed.")
        for idx, row in df_params.iterrows(): # iterate over masks
            seg_fname = os.path.join(args.seg_path, row["mask_name"])
            if not args.tiff_output:
                seg_fname = seg_fname + ".hdf5"
                os.remove(seg_fname)
            else:
                rmtree(seg_fname)
        
    return

if __name__ == "__main__":

    parser = configargparse.ArgParser()
    
    # Setup segmenter with i/o inputs
    parser.add('-c', '--config-setupseg', required=True, is_config_file=True, help='config file for segmenter')
    parser.add('-p', '--ct_fpath', required = True, type = str, help = 'path to CT dataset (.hdf5 only)')
    parser.add('--ct_data_tag', required = True, type = str, help = 'dataset name/path in hdf5 file')
    parser.add('--seg_path', required = True, type = str, help = 'path to folder for saving masks (may not exist)')
    parser.add('--model_path', required = True, type = str, help = 'path to parent folder containing model file (model repository)')
    parser.add('-m', '--model_name', required = True, type = str, help = 'model name')
    parser.add('-v', '--vote_maskname', required = False, type = str, default = 'VOTED', help = 'name of final (voted) mask')
    parser.add('--remove_masks', required = False, type = str_to_bool, default = False, metavar = 'bool', help = 'True to delete all intermediate masks')
    parser.add('--run_ensemble', required = False, type = str_to_bool, default = True, metavar = 'bool', help = 'True to run ensemble vote on all masks')
    parser.add('--run_seg', required = False, type = str_to_bool, default = True, metavar = 'bool', help = 'True to generate intermediate segmentation maps. If False, only voter will run')
    parser.add('--stats_only', required = False, type = str_to_bool, default = False, metavar = 'bool', help = 'if True, program will exit after showing dataset stats')
    parser.add('--mem_thres', required = False, type = float, default = 1.0, metavar = 'float (GB)', help = 'amount of data (GB) to be read from CT data at a time')
    parser.add('--overwrite_OK', required = False, type = str_to_bool, default = False, metavar = 'bool', help = 'if mask name already exists, is overwrite OK?')
    parser.add('--rw_verbosity', required = False, type = int, default = 1, metavar = 'integer', help = 'Read/Write verbosity 0 - silent, 1 - import stuff, 2 - everything')
    parser.add('--tiff_output', required = False, type = str_to_bool, default = True, metavar = 'bool', help = 'True to save final (ensemble vote) mask as tiff sequence in folder')
    parser.add('--nprocs', required = False, type = int, default = 1, metavar = 'integer', help = 'use these many processors on each subset of chunk read into memory')
    parser.add('--arr_split', required = False, type = int, default = 1, metavar = 'integer', help = 'break down chunk read in memory into these many subsets for processing')
    parser.add('--mpl_agg', required = False, type = str, default = 'Agg', help = 'matplotlib backend to use')
    
    parser.add('--mask_name', action = 'append', type = str, default = [])
    parser.add('--n_patches', type = n_patches_type, action = 'append') #ast.literal_eval
    parser.add('--slice_axis', type = int, action = 'append')
    parser.add('--overlap', type = int, action = 'append')
    parser.add('--rotation', type = float, action = 'append')
    parser.add('--crops', type = crops_type, action = 'append')
    args = parser.parse_args()

#     print(parser.format_values())
#     print(args)
    
    main(args)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        