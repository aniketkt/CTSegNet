#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 17:13:22 2019
@author: atekawade

"""
import os
import numpy as np
import pandas as pd # line 13 empty for good luck

import shutil
import h5py
import glob

import matplotlib.pyplot as plt
import matplotlib as mpl

from tkinter import *
from tkinter import filedialog as fd
from matplotlib.patches import Rectangle

import functools
from multiprocessing import cpu_count, Pool
from skimage.io import imread


DEBUG_MODE = True

def _message(_str, bool_in):
    if bool_in:
        print(_str)
        
        
def browse_path(_msg, default_path = "/"):

    root = Tk()
    root.withdraw()
    _data_path = fd.askopenfilename(parent = root, initialdir = default_path, title = _msg + "Can be TIFF directory or HDF5.", filetypes = (('tiff file series', ('*.tif','*.tiff')),('hdf5 file', '*.hdf5')))
    if ".tif" in _data_path:
        _data_path = "/" + os.path.join(*_data_path.split('/')[:-1])
        _tiff = True
    else:
        _tiff = False
        
    root.destroy()
    return _data_path, _tiff
        

def browse_file(_msg, f_type = ("comma delimited", "*.csv")):

    root = Tk()
    root.withdraw()
    _data_path = fd.askopenfilename(parent = root, title = _msg, filetypes = (f_type,))
        
    root.destroy()
    return _data_path


def browse_files(_msg, f_type = ("comma delimited", "*.csv")):

    root = Tk()
    root.withdraw()
    _data_paths = fd.askopenfilenames(parent = root, title = _msg, filetypes = (f_type,))
        
    root.destroy()
    return _data_paths



def browse_savepath(_msg):

    root = Tk()
    root.withdraw()
    _data_path = fd.askdirectory(parent = root, title = _msg)
        
    root.destroy()
    return _data_path


def handle_YN(_str):
    
    _inp = input(_str)
    if _inp in ("yes", "Yes", "Y", "y"):
        return True
    elif _inp in ("no", "No", "N", "n"):
        return False
    else:
        raise InputError("Input not understood")
    return

class InputError(Exception):
    def __init__(self, message):
        self.message = message


class DataFile():
    def __init__(self, fname, data_tag = None, tiff = False,\
                 chunked_slice_size = None, d_shape = None, \
                 d_type = None, VERBOSITY = 1):
        """An instance of a DataFile class is a 3D dataset in a tiff sequence or hdf5 file.
        Methods are provided to read chunks and slices.
        fname     : path to hdf5 filename or folder containing tiff sequence
        tiff      : bool, True if fname is path to tiff sequence, else False
        data_tag  : str, dataset name / path in hdf5 file. None if tiff sequence
        VERBOSITY : int 0 - print nothing, 1 - important stuff, 2 - print everything
        optionals (required for non-existent dataset only) -->
        d_shape   : tuple, shape of dataset 
        d_type    : np.<dtype>
        chunked_slice_size : float, size in GB for setting chunk shape in hdf5
        """
        self.fname = fname
        self.data_tag = data_tag # for hdf5 only
        self.VERBOSITY = VERBOSITY
        
        self.tiff_mode = tiff
        self.exists = os.path.exists(self.fname)
        self.chunked_slice_size = chunked_slice_size

        if (d_type is not None) & (d_shape is not None):
            self.d_type = d_type # overwrite previously read stats            
            self.d_shape = d_shape              
            self.get_slice_sizes()
            self.est_chunking()   
            _message("\n" + "#"*50 + "\n" + "New dataset will be created in %s: "%("tiff folder" if self.tiff_mode else "hdf5 file") + self.fname.split('/')[-1], self.VERBOSITY > 1)        
        elif self.exists:
            self.get_stats()
            self.get_slice_sizes()

        else:
            raise ValueError("Inputs d_shape and d_type are required for new dataset as file does not already exist.\nPath does not exist:\n%s"%fname)
            
        return
    
    def set_verbosity(self, val):
        self.VERBOSITY = val
        return

    def create_new(self, overwrite = False):
        if self.tiff_mode:
            if os.path.exists(self.fname):
                if overwrite:
                    shutil.rmtree(self.fname)
                    _message("Removed old contents in tiff folder %s"%(self.fname.split('/')[-1]), self.VERBOSITY > 0)
                else:
                    raise ValueError("tiff folder already exists and overwrite is not allowed.")
        else:
            if self.data_tag is None:
                raise ValueError("data_tag: argument missing for creating new hdf5 file.")
            if os.path.exists(self.fname):
                with h5py.File(self.fname, 'r') as hf:
                    if self.data_tag in hf.keys():
                        if overwrite:
                            os.remove(self.fname)
                            _message("Removed old hdf5 file %s"%(self.fname.split('/')[-1]), self.VERBOSITY > 0)
                        else:
                            raise ValueError("hdf5 dataset already exists and overwrite is not allowed.")
                    else:
                        pass

            self.est_chunking()
            _opt = 'r+' if os.path.exists(self.fname) else 'w'
            with h5py.File(self.fname, _opt) as hf:
                hf.create_dataset(self.data_tag, shape = self.d_shape, dtype = self.d_type, chunks = self.chunk_shape)
            _message("New hdf5 dataset %s created in file %s"%(self.data_tag, self.fname.split('/')[-1]), self.VERBOSITY > 0)
            
    def get_stats(self):
    
        if not self.tiff_mode:
            with h5py.File(self.fname, 'r') as hf:
                self.d_shape = hf[self.data_tag].shape
                self.d_type = hf[self.data_tag].dtype
                self.chunk_shape = hf[self.data_tag].chunks
        elif self.tiff_mode:
            img_path = glob.glob(self.fname+'/*.tif')
            if not img_path: img_path = glob.glob(self.fname+'/*.tiff')
            img_z = len(img_path)
            img = imread(img_path[0])
            self.d_shape = (img_z,) + img.shape
            self.d_type = img.dtype
            self.chunk_shape = None # no such attribute for tiff stacks
            
        _message("\n" + "#"*50 + "\n" + "Found existing %s: "%("tiff folder" if self.tiff_mode else "hdf5 file") + self.fname.split('/')[-1], self.VERBOSITY > 0)        
        _message("Dataset shape: %s"%(str(self.d_shape)), self.VERBOSITY > 0)
            
        return

    def get_slice_sizes(self):
        if self.d_type == np.float32:
            fac = 4.0
        elif self.d_type == np.uint8:
            fac = 1.0
        elif self.d_type == np.uint16:
            fac = 2.0
        else:
            raise ValueError("Data type %s is not supported."%self.d_type)

        self.slice_size = (np.prod(self.d_shape[1:])*fac/(1e9), np.prod(self.d_shape[::2])*fac/(1e9), np.prod(self.d_shape[:-1])*fac/(1e9))

        return


    def show_stats(self):

        _message("Dataset shape: %s"%(str(self.d_shape)), self.VERBOSITY > -1)
        if not self.tiff_mode: _message("Chunk shape: %s"%(str(self.chunk_shape)), self.VERBOSITY > -1)
        for _i, _size in enumerate(self.slice_size):
            _message("Slice size along %i: %4.2f MB"%(_i, 1000.0*_size), self.VERBOSITY > -1)
        
    def est_chunking(self): # Determine the chunk shape for hdf5 file, optimized for slicing along all 3 axes
        # max_slice_size : in GB
        
        if self.tiff_mode:
            self.chunked_shape = None
        else:
            if self.chunked_slice_size is None:
                self.chunk_shape = None
            else:
                self.chunk_shape = tuple(int(np.ceil(self.chunked_slice_size / single_slice_size)) for single_slice_size in self.slice_size)
            _message("Estimated Chunk shape: %s"%str(self.chunk_shape), self.VERBOSITY > 1)
        return

    def read_slice(self, axis = None, slice_idx = None):
        
        img, s = self.read_chunk(axis = axis, slice_start = slice_idx, slice_end = slice_idx + 1)
        return img[0]

    def read_data(self, slice_3D = (slice(None,None),)*3):
        """Read a block of data. Only supported for hdf5 datasets.
        slice_3D   : list of three python slices e.g. [slice(start,stop,step)]*3
        """
        
        if self.tiff_mode:
            ch = np.asarray(read_tiffseq(self.fname, s = slice_3D[0]))
            ch = ch[:, slice_3D[1], slice_3D[2]]
        
        with h5py.File(self.fname, 'r') as hf:
            
            _message("Reading hdf5: %s, Z: %s, Y: %s, X: %s"%(self.fname.split('/')[-1], str(slice_3D[0]), str(slice_3D[1]), str(slice_3D[2])), self.VERBOSITY > 1)    
            ch = np.asarray(hf[self.data_tag][slice_3D[0],slice_3D[1],slice_3D[2]])
        return ch

    def read_chunk(self, axis = None, slice_start = None, chunk_shape = None, max_GB = 10.0, slice_end = "", skip_fac = None):
        """Read a chunk of data along a given axis.
        axis         : int in (0,1,2), axis > 0 is not supported for tiff series
        slice_start  : int, start index along axis
        chunk_shape  : tuple, (optional) used if hdf5 has no attribute chunk_shape
        max_GB       : float, maximum size of chunk that's read. slice_end will be calculated from this.
        slice_end    : int, (optional) used if max_GB is not provided.
        skip_fac     : int, (optional) "step" value as in slice(start, stop, step)
        """
        if slice_end == "":
            # Do this to determine slice_end based on a max_GB value as RAM limit
            if (chunk_shape is None) & (not self.tiff_mode):
                chunk_shape = self.chunk_shape
            chunk_len = chunk_shape[axis] if chunk_shape is not None else 1
            slice_len = max(1,int(np.round(max_GB/self.slice_size[axis])//chunk_len))*chunk_len
            slice_end =  min(slice_len + slice_start , self.d_shape[axis])
        elif slice_end is None:
            slice_end = self.d_shape[axis]
        
        s = slice(slice_start, slice_end, skip_fac)
        
        if not self.tiff_mode: # hdf5 mode
            with h5py.File(self.fname, 'r') as hf:
                
                _message("Reading hdf5: %s, axis %i,  %s, with chunk_shape: %s"%(self.fname.split('/')[-1], \
                                                                                 axis, s, chunk_shape), self.VERBOSITY > 1)    
                if axis == 0:
                    ch = np.asarray(hf[self.data_tag][s,:,:])
                elif axis == 1:
                    ch = np.asarray(hf[self.data_tag][:,s,:])
                elif axis == 2:
                    ch = np.asarray(hf[self.data_tag][:,:,s])
                ch = np.moveaxis(ch, axis, 0)
            return ch, s
        
        else: # in tiff mode
            if axis != 0: raise ValueError("TIFF data format does not support multi-axial slicing.")

            _message("Reading tiff: %s, axis %i,  %s, chunk_shape: %s"%(self.fname.split('/')[-1],\
                                                                        axis, s, chunk_shape), self.VERBOSITY > 1)    
            ch = np.asarray(read_tiffseq(self.fname, s = s))
            return ch, s

    def read_full(self):
        """Read the full dataset
        """
        
        ch, s = self.read_chunk(axis = 0, slice_start = 0, slice_end = self.d_shape[0], skip_fac = None)
        return ch
    
    def write_full(self, ch):
        """Write the full dataset to filepath.
        """
        self.write_chunk(ch, axis = 0, s = slice(0, self.d_shape[0]))
        return        

    def write_data(self, ch, slice_3D = None):
        """Write a block of data. Only supported for hdf5 datasets.
        ch         : numpy array (3D), to be written
        slice_3D   : list of three python slices e.g. [slice(start,stop,step)]*3 - must match shape of ch
        """
        
        if self.tiff_mode:
            raise ValueError("Not supported for tiff data")

        with h5py.File(self.fname, 'r+') as hf:
    
            _message("Saving hdf5: %s, Z: %s, Y: %s, X: %s"%(self.fname.split('/')[-1], str(slice_3D[0]), str(slice_3D[1]), str(slice_3D[2])), self.VERBOSITY > 1)    

            hf[self.data_tag][slice_3D[0],slice_3D[1],slice_3D[2]] = ch
            _message("Done", self.VERBOSITY > 1)
        return
        
        
    def write_chunk(self, ch, axis = None, s = None):
        """Write a chunk of data along a given axis.
        axis  : int in (0,1,2), axis > 0 is not supported for tiff series
        s     : python slice(start, stop, step) - step must be None for tiff series
        """
        if not self.tiff_mode:
                
            if not os.path.exists(self.fname):
                raise ValueError("hdf5 file needs to exist before writing chunks.")
            
            with h5py.File(self.fname, 'r+') as hf:
        
                _message("Saving %s, axis %i,  slice %i to %i"%(self.fname.split('/')[-1], axis, s.start, s.stop), self.VERBOSITY > 1)    

                ch = np.moveaxis(ch, 0, axis)
                if axis == 0:
                    hf[self.data_tag][s,:,:] = ch
                elif axis == 1:
                    hf[self.data_tag][:,s,:] = ch
                elif axis == 2:
                    hf[self.data_tag][:,:,s] = ch
                _message("Done", self.VERBOSITY > 1)
    
        else:
            if axis != 0:
                raise ValueError("TIFF data format does not support multi-axial slicing.")
            if os.path.exists(self.fname) & (s.start == 0):
                raise FileExistsError("TIFF folder to be written already exists. Please delete first.")
            _message("Saving %s, axis %i,  slice %i to %i"%(self.fname.split('/')[-1], axis, s.start, s.stop), self.VERBOSITY > 1)    
            
            write_tiffseq(ch, SaveDir = self.fname, increment_flag = True,\
                          suffix_len = len(str(self.d_shape[axis])),\
                          dtype = self.d_type)
        
        return
    
from scipy.ndimage.filters import median_filter
def get_domain_extent(d, min_size = 512):

    while True:
        fig, ax = plt.subplots(1, 2, figsize = (20,10))

        ax[0].imshow(d[d.shape[0]//2,...], cmap = 'Greys', alpha = 0.5)
        ax[0].set_title("First Select two points to define a rectangle")
        plt.pause(0.1)
        pts = np.asarray(plt.ginput(2))
        xy = pts[0,:]
        w, h = pts[1,:] - pts[0,:]
        rect = Rectangle(xy, w, h, fill = False)
        ax[0].add_patch(rect)
        plt.pause(0.05)
        crop_X, crop_Y = np.sort(pts[:,0]).astype(np.uint16), np.sort(pts[:,1].astype(np.uint16))
        
        show_img = median_filter(d[:,d.shape[1]//2,:], size = 5)
        ax[1].imshow(show_img, cmap = 'Greys', alpha = 0.5)
        ax[1].set_title("Then Select the top and bottom extents of the domain.")
        plt.pause(0.05)
        pts = np.asarray(plt.ginput(2))[:,1]
        xy = (crop_X[0], pts.min())
        w = crop_X[1] - crop_X[0]
        h = pts.max() - pts.min()
        rect = Rectangle(xy, w, h, fill = False)
        ax[1].add_patch(rect)
        plt.pause(0.05)
        crop_Z = np.sort(pts.astype(np.uint16))

        if (crop_X[1] - crop_X[0] < min_size) or (crop_Y[1] - crop_Y[0] < min_size) or (crop_Z[1] - crop_Z[0] < min_size):
            print("Domain extents must be larger than model patch size")
            plt.close()
            continue
    
        if handle_YN("Happy?" ):
            break
        else:
            plt.close()
    
    return crop_Z, crop_Y, crop_X

def read_tiffseq(userfilepath = '', procs = None, s = None):
    """Read a sequence of tiff images from folder.
    userfilepath : path to folder containing images
    s            : s is either a slice(start, stop, step) or a list of indices to be read  
    """
    if not userfilepath:
        raise ValueError("File path is required.")
        return []

    if procs == None:
        procs = cpu_count()
        
    ImgFileList = sorted(glob.glob(userfilepath+'/*.tif'))
    
    if not ImgFileList: ImgFileList = sorted(glob.glob(userfilepath+'/*.tiff'))
    
    if s != None:
        if type(s) == slice:
            ImgFileList = ImgFileList[s]
        elif type(s) == list:
            ImgFileList = [ImgFileList[i] for i in s]
        else:
            raise ValueError("s input not recognized.")
    
    Im_Stack = Parallelize(ImgFileList, imread, procs = procs)

    return Im_Stack

def write_tiffseq(Im_Stack, SaveDir = "", increment_flag = False,\
                  suffix_len = None):
    """Write a sequence of tiff images to a directory.
    S              : numpy array (3D), sequence will be created along axis 0
    SaveDir        : str, path to folder, will create directory if doesn't exist
    increment_flag : bool, True to write append images to existing ones in folder
    suffix_len     : int, e.g. 4 for 1000 images, 5 for 10,000
    dtype          : np.<dtype> - data will be typecast to this!
    """
    if not suffix_len:
        if increment_flag:
            raise ValueError("suffix_len required if increment_flag is True.")
        else:
            suffix_len = len(str(S.shape[0]))
    
    last_num = 0
    if not os.path.exists(SaveDir):
        os.makedirs(SaveDir)
    else:
        if not increment_flag:
            shutil.rmtree(SaveDir)
            os.makedirs(SaveDir)
        else:
            ImgFileList = sorted(glob.glob(SaveDir+'/*.tif'))
            if not ImgFileList: ImgFileList = sorted(glob.glob(SaveDir+'/*.tiff'))
            if not ImgFileList:
                last_num = 0
            else:
                last_num = int(ImgFileList[-1].split('.')[0][-suffix_len:])    
    
    BaseDirName = os.path.basename(os.path.normpath(SaveDir))
    
    for iS in range(S.shape[0]):
        img_num = str(iS+1+last_num).zfill(suffix_len)
        imsave(os.path.join(SaveDir, BaseDirName + img_num + '.tif'), S[iS])
    
    return
    
def Parallelize(ListIn, f, procs = -1, **kwargs):
    
    """This function packages the "starmap" function in multiprocessing, to allow multiple iterable inputs for the parallelized function.
    ListIn : list, each item in the list is a tuple of non-keyworded arguments for f.
    f      : function to be parallelized. Signature must not contain any other non-keyworded arguments other than those passed as iterables.
    # Example:
    # def multiply(x, y, factor = 1.0):
    #   return factor*x*y
    # X = np.linspace(0,1,1000)
    # Y = np.linspace(1,2,1000)
    # XY = [ (x, Y[i]) for i, x in enumerate(X)] # List of tuples
    # Z = Parallelize_MultiIn(XY, multiply, factor = 3.0, procs = 8)
    # Create as many positional arguments as required, but remember all must be packed into a list of tuples.
    """
    if type(ListIn[0]) != tuple:
        ListIn = [(ListIn[i],) for i in range(len(ListIn))]
    
    reduced_argfunc = functools.partial(f, **kwargs)
    
    if procs == -1:
        opt_procs = int(np.interp(len(ListIn), [1,100,500,1000,3000,5000,10000] ,[1,2,4,8,12,36,48]))
        procs = min(opt_procs, cpu_count())

    if procs == 1:
        OutList = [reduced_argfunc(*ListIn[iS]) for iS in range(len(ListIn))]
    else:
        p = Pool(processes = procs)
        OutList = p.starmap(reduced_argfunc, ListIn)
        p.close()
        p.join()
    
    return OutList
   

if __name__ == "__main__":
    
    mem_thres = 20.0
    _message("Data input / output functions.")



