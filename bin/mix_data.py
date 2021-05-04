#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 20:36:02 2020

@author: atekawade
"""
import os
import glob
from ct_segnet.data_utils.data_augmenter import remove_blanks
from ct_segnet.data_utils.data_io import DataFile
import numpy as np
import argparse
import time
import pandas as pd



def main(args):

    data_path = args.data_path
    csv_path = args.csv_path
    df_map = pd.read_csv(csv_path)
    
    Y = []
    X = []


    for idx, row in df_map.iterrows():

        ds_X = DataFile(os.path.join(data_path, row["fname"]),\
                        data_tag = 'X', VERBOSITY = 0, tiff = False)
        ds_Y = DataFile(os.path.join(data_path, row["fname"]), \
                        data_tag = 'Y', VERBOSITY = 0, tiff = False)
        
        y_train = ds_Y.read_full()
        x_train = ds_X.read_full()
        

        if not args.retain_blanks:
            print("before: %i"%len(x_train))
            x_train, y_train = remove_blanks(x_train, y_train, cutoff = 0.2)
            print("after remove blanks: %i"%len(x_train))

        idxs =  np.random.choice(len(x_train), \
                                 size = row["sample_size"], \
                                 replace=False)
        
        x_train = x_train[idxs,...]
        y_train = y_train[idxs,...]
        
        X.append(x_train)
        Y.append(y_train)

        print("Done %s, images %i"%(row['fname'], len(y_train)))

    Y = np.concatenate(Y, axis = 0)
    X = np.concatenate(X, axis = 0)


    # save stuff
    dsX = DataFile(os.path.join(data_path, args.out_fname.split('.')[0] + '.hdf5'), \
                                data_tag = 'X', \
                                d_shape = X.shape, \
                                d_type = np.float32, \
                                VERBOSITY = 0)
    dsX.create_new(overwrite = args.overwrite_OK)
    dsY = DataFile(os.path.join(data_path, args.out_fname.split('.')[0] + '.hdf5'), \
                                data_tag = 'Y', \
                                d_shape = Y.shape, \
                                d_type = np.uint8, \
                                VERBOSITY = 0)
    dsY.create_new(overwrite = True)

    dsX.write_full(X)
    dsY.write_full(Y)    
    



    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Mix training data from multiple datsets')
    
    
    parser.add_argument('-p', '--data_path', required = True, type = str, help = 'path to parent folder containing all data sets')
    parser.add_argument('-f', '--csv_path', required = True, type = str, help = 'path to csv file listing data sets and respective sample size')
    parser.add_argument('-o', '--out_fname', required = True, type = str, help = 'output hdf5 filename')
    parser.add_argument('-b', '--retain_blanks', required = False, action = 'store_true', default = False, help = 'if true, do not remove blank images from training data')    
    parser.add_argument('-w', "--overwrite_OK", required = False, action = "store_true", default = False, help = "if output file exists, overwrite")
#
    args = parser.parse_args()
    
    time_script_start = time.time()
    main(args)
    time_script_end = time.time()
    tot_time_script = (time_script_end - time_script_start)/60.0
    
    _str = "Total time elapsed for script: %.2f minutes"%tot_time_script 

    
    
    
    
    
