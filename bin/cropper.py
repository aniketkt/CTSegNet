#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crop a dataset with given input argument 'crops'. Data will always be saved as tiff.

@author: atekawade
"""

import numpy as np
import os
import sys
from ct_segnet.data_utils import data_io

import argparse
from argparse import ArgumentTypeError
from tqdm import tqdm
import ast
from shutil import rmtree
import time

from ct_segnet.data_utils.data_io import str_to_bool, n_patches_type, crops_type
mem_thres = 0.0

def main(args):
    data_io.show_header()
    # Understand input data format
    if os.path.isdir(args.input_fname):
        tiff_input = True
        if args.dataset_name == "": args.dataset_name = "data"
    elif args.input_fname.split('.')[-1] in ("hdf5", "h5"):
        tiff_input = False
        if args.dataset_name == "": raise ArgumentTypeError("dataset-name required for hdf5")
    else:
        raise ArgumentTypeError("input file type not recognized. must be tiff folder or hdf5 file")
    input_fname = args.input_fname
    
    # if output_fpath is not provided, then get parent path to input file
    if args.output_fpath == "":
        args.output_fpath = os.path.split(args.input_fname)[0]
        
    # set output filename - automatically set to fname + 'CROPPED' if none is provided
    if args.output_fname == "cropped_tiff":
        args.output_fname = os.path.split(args.input_fname)[-1].split('.')[0] + '_CROPPED'
    else:
        args.output_fname = args.output_fname.split('.')[0]
    output_fname = os.path.join(args.output_fpath, args.output_fname)    
    
        
    # Define DataFile instances
    r_dfile = data_io.DataFile(input_fname, tiff = tiff_input, \
                               data_tag = args.dataset_name, \
                               VERBOSITY = args.verbosity)

    print("Input data stats:")
    r_dfile.show_stats()
    if args.stats_only:
        sys.exit()
    
    str_prompt = "\nData will be saved to the following location.\n%s"%output_fname
    if not args.yes:
        input(str_prompt + "\nPress any key to continue")
    else:
        print(str_prompt)
    
    t0 = time.time()
    vol = r_dfile.read_full()
    
    vol = vol[slice(*args.cropz), slice(*args.cropy), slice(*args.cropx)]
    
    
    w_shape = vol.shape # future implementation must allow resampling dataset
    w_dtype = r_dfile.d_type # future implementation must allow changing dtype (with renormalization)
    
    w_dfile = data_io.DataFile(output_fname, tiff = True, \
                               VERBOSITY = args.verbosity, \
                               d_shape = w_shape, d_type = w_dtype)
                               
    w_dfile.create_new(overwrite = args.overwrite_OK)
    w_dfile.write_full(vol)
    
    total_time = (time.time() - t0)/60.0 # minutes
    print("\nTotal time: %.2f minutes"%(total_time))
    
    if args.delete:
        
        if not args.yes: input("Delete old file? Press any key")
        if tiff_input:
            rmtree(input_fname)
        else:
            os.remove(input_fname)
        
if __name__ == "__main__":
    
    # Arg parser stuff
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', "--input-fname", required = True, type = str, help = "Path to tiff folder or hdf5 file")
    parser.add_argument('-i', "--stats_only", required = False, action = "store_true", default = False, help = "show stats only")
    parser.add_argument('-d', "--delete", required = False, action = "store_true", default = False, help = "delete input file")
    parser.add_argument('-o', "--output-fpath", required = False, default = "", type = str, help = "Parent folder path for saving output. (optional) If not provided, file will be written to same parent folder as input file.")
    parser.add_argument('-n', "--output-fname", required = False, default = "cropped_tiff", type = str, help = "Name of output hdf5 file.")
    parser.add_argument('-x', "--dataset-name", required = False, type = str, default = "", help = "Dataset name for hdf5; required if input is hdf5 file")
    parser.add_argument('-v', "--verbosity", required = False, type = int, default = 0, help = "read / write verbosity; 0 - silent, 1 - important stuff, 2 - print everything")
    parser.add_argument('-w', "--overwrite_OK", required = False, action = "store_true", default = False, help = "if output file exists, overwrite")
#     parser.add_argument('-r', "--resample-factor", required = False, type = int, help = "resample to reduce dataset size by cube of this factor")

    parser.add_argument('--cropz', type = crops_type, default = '(None, None)', action = 'store', required = False)
    parser.add_argument('--cropy', type = crops_type, default = '(None, None)', action = 'store', required = False)
    parser.add_argument('--cropx', type = crops_type, default = '(None, None)', action = 'store', required = False)
    
    parser.add_argument('-y', "--yes", required = False, action = "store_true", default = False, help = "say yes to all prompts")
    args = parser.parse_args()
    main(args)
    
    
    
    
