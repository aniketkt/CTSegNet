#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A memory-efficient interface to slice, read and write CT data. Tiff series and hdf5 data formats are currently supported.
"""
import os
import numpy as np
import pandas as pd # line 13 empty for good luck

import shutil
import h5py
import glob

import matplotlib.pyplot as plt
import matplotlib as mpl

import functools
from multiprocessing import cpu_count, Pool
from skimage.io import imread
from tifffile import imsave
from configargparse import ArgumentTypeError
import ast


DEBUG_MODE = True

def _message(_str, bool_in):
    """
    :meta private:
    """
    
    if bool_in:
        print(_str)
        
def handle_YN(_str):
    """
    :meta private:
    """
    _inp = input(_str)
    return str_to_bool(_inp)


def str_to_bool(_inp):
    """
    
    :meta private:
    
    """
    _inp = str(_inp)
    if _inp in ("yes", "Yes", "Y", "y", "True", "TRUE", "true"):
        return True
    elif _inp in ("no", "No", "N", "n", "False", "FALSE", "false"):
        return False
    else:
        raise ArgumentTypeError("Input not understood")
    return

def n_patches_type(s):
    """
    :meta private:
    """
    s = s.split('x')
    s = ','.join(s)
    return ast.literal_eval(s)

def crops_type(s):
    """
    :meta private:
    """
    s = s.split(':')
    s = ','.join(s)
    return ast.literal_eval(s)


class InputError(Exception):
    """
    :meta private:
    """
    def __init__(self, message):
        self.message = message


class DataFile():
    """An instance of a DataFile class points to a 3D dataset in a tiff sequence or hdf5 file. The interface includes read/write methods to retrieve the data in several ways (slices, chunks, down-sampled data, etc.)  
    
    For setting chunk size in hdf5, either chunk_shape > chunk_size > chunked_slice_size can be input. If two or more are provided, this order is used to select one.  
    
    Parameters
    ----------
    fname : str
        path to hdf5 filename or folder containing tiff sequence  

    tiff : bool
        True if fname is path to tiff sequence, else False  
    
    data_tag: str
        dataset name / path in hdf5 file. None if tiff sequence  
    
    VERBOSITY : int
        0 - print nothing, 1 - important stuff, 2 - print everything  
    
    d_shape : tuple
        shape of dataset; required for non-existent dataset only  
    
    d_type : numpy.dtype
        data type for voxel data; required for non-existent dataset only  
    
    chunk_size: float
        in GB - size of a hyperslab of shape proportional to data shape    
    
    chunked_slice_size : float
        in GB - size of a chunk of some slices along an axis  
    
    chunk_shape : tuple
        shape of hyperslab for hdf5 chunking  

    Example  
    
    .. highlight:: python   
    .. code-block:: python  
    
        from ct_segnet.data_io import DataFile  
        # If fname points to existing hdf5 file  
        dfile = DataFile(fname, tiff = False, data_tag = "dataset_name")

        # read a slice  
        img = dfile.read_slice(axis = 1, slice_idx = 100)  

        # read a chunk  of size 2.0 GB starting at slice_start = 0
        vol, s = dfile.read_chunk(axis = 1, slice_start = 0, max_GB = 2.0)  

        # read a chunk between indices [10, 100], [20, 200], [30, 300] along the respective axes  
        vol = dfile.read_data(slice_3D = [slice(10, 100), slice(20, 200), slice(30,300)])

        # or just read all the data
        vol = dfile.read_full()

    """
    
    def __init__(self, fname, data_tag = None, tiff = False,\
                 chunk_shape = None, chunk_size = None, chunked_slice_size = None,\
                 d_shape = None, d_type = None, VERBOSITY = 1):
        self.fname = fname
        self.data_tag = data_tag # for hdf5 only
        self.VERBOSITY = VERBOSITY
        
        self.tiff_mode = tiff
        self.exists = os.path.exists(self.fname)
        self.chunked_slice_size = chunked_slice_size
        self.chunk_size = chunk_size
        self.chunk_shape = chunk_shape

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
    
    def set_verbosity(self, VERBOSITY):
        """
        """
        self.VERBOSITY = val
        return

    def create_new(self, overwrite = False):
        """
        For hdf5 - creates an empty dataset in hdf5 and assigns shape, chunk_shape, etc. For tiff folder - checks if there is existing data in folder.  
        
        :param bool overwrite: if True, remove existing data in the path (fname).
        
        """
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
            
    def get_stats(self, return_output = False):
        """
        Print some stats about the DataFile (shape, slice size, chunking, etc.)  
        
        """
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
        
        
        str_out = "\n" + "#"*50 + "\n" + "Found existing %s: "%("tiff folder" if self.tiff_mode else "hdf5 file") + self.fname.split('/')[-1]
        str_out = str_out + "\nDataset shape: %s"%(str(self.d_shape))
        
        if return_output:
            return str_out
        else:
            _message(str_out, self.VERBOSITY > 0)        
            
        return

    def get_slice_sizes(self):
        """  
        """
        if self.d_type == np.float32:
            fac = 4.0
        elif self.d_type == np.uint8:
            fac = 1.0
        elif self.d_type == np.uint16:
            fac = 2.0
        else:
            raise ValueError("Data type %s is not supported."%self.d_type)

        self.slice_size = (np.prod(self.d_shape[1:])*fac/(1e9), np.prod(self.d_shape[::2])*fac/(1e9), np.prod(self.d_shape[:-1])*fac/(1e9))
        self._bytes_per_voxel = fac
        self.d_size_GB = fac*np.prod(self.d_shape)/1e9
        return


    def show_stats(self, return_output = False):
        """print dataset shape and slice-wise size
        """
        str_out = ""
        str_out = str_out + "\n" + "Dataset shape: %s"%(str(self.d_shape))
        str_out = str_out + "\n" + "Dataset size: %.2f GB"%self.d_size_GB
        
        if not self.tiff_mode:
            str_out = str_out + "\n" + "Chunk shape: %s"%(str(self.chunk_shape))
            
        for _i, _size in enumerate(self.slice_size):
            str_out = str_out + "\n" + "Slice size along %i: %4.2f MB"%(_i, 1000.0*_size)
        
        if return_output:
            return str_out
        else:
            _message(str_out, self.VERBOSITY > -1)
            return

        
    def est_chunking(self): # Determine the chunk shape for hdf5 file, optimized for slicing along all 3 axes
        """  
        """
        
        if self.tiff_mode:
            self.chunked_shape = None
        else:
            if self.chunk_shape is not None:
                return # allow user to define chunk_shape
            if self.chunk_size is not None:
                fac = np.cbrt((self.chunk_size) / (self.d_size_GB))
                self.chunk_shape = tuple([int(self.d_shape[i]*fac) for i in range(3)])
                return
            if self.chunked_slice_size is not None:
                if self.chunked_slice_size > (self.slice_size[0]*self.d_shape[0]):
                    raise ValueError("chunked_slice_size cannot be larger than dataset size")
                self.chunk_shape = tuple(int(np.ceil(self.chunked_slice_size / single_slice_size)) for single_slice_size in self.slice_size)
                _message("Estimated Chunk shape: %s"%str(self.chunk_shape), self.VERBOSITY > 1)
                return
            else:
                self.chunk_shape = None
                return
        return

    def read_slice(self, axis = None, slice_idx = None):
        """Read a slice.  
        
        :param int axis: axis 0, 1 or 2  
        
        :param int slice_idx: index of slice along given axis  
        
        """
        img, s = self.read_chunk(axis = axis, slice_start = slice_idx, slice_end = slice_idx + 1)
        return img[0]

    def read_data(self, slice_3D = (slice(None,None),)*3):
        """Read a block of data. Only supported for hdf5 datasets.  
        
        :param list slice_3D: list of three python slices e.g. [slice(start,stop,step)]*3
        
        """
        
        if self.tiff_mode:
            ch = np.asarray(read_tiffseq(self.fname, s = slice_3D[0]))
            ch = ch[:, slice_3D[1], slice_3D[2]]
        
        with h5py.File(self.fname, 'r') as hf:
            
            _message("Reading hdf5: %s, Z: %s, Y: %s, X: %s"%(self.fname.split('/')[-1], str(slice_3D[0]), str(slice_3D[1]), str(slice_3D[2])), self.VERBOSITY > 1)    
            ch = np.asarray(hf[self.data_tag][slice_3D[0],slice_3D[1],slice_3D[2]])
        return ch

    def read_sequence(self, idxs):
        """Read a list of indices idxs along axis 0.  
        
        :param int axis: axis 0, 1 or 2  
        
        :param list idxs: list of indices  

        """
        with h5py.File(self.fname, 'r') as hf:
            return np.asarray(hf[self.data_tag][idxs,...])
    
    def read_chunk(self, axis = None, slice_start = None, chunk_shape = None, max_GB = 10.0, slice_end = "", skip_fac = None):
        """Read a chunk of data along a given axis.  
        
        Parameters
        ----------
        
        axis : int
            axis > 0 is not supported for tiff series  
        
        slice_start : int
            start index along axis  
        
        chunk_shape : tuple
            (optional) used if hdf5 has no attribute chunk_shape  
        
        max_GB : float
            maximum size of chunk that's read. slice_end will be calculated from this.  
        
        slice_end : int
            (optional) used if max_GB is not provided.  
        
        skip_fac : int
            (optional) "step" value as in slice(start, stop, step)  
        
        Returns
        -------
        tuple
            (data, slice) where data is a 3D numpy array  
        
        
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

    def read_full(self, skip_fac = None):
        """Read the full dataset
        """
        
        ch, s = self.read_chunk(axis = 0, slice_start = 0, slice_end = self.d_shape[0], skip_fac = skip_fac)
        return ch
    
    def write_full(self, ch):
        """Write the full dataset to filepath.  
        
        :param ch: 3D numpy array to be saved
        
        """
        self.write_chunk(ch, axis = 0, s = slice(0, self.d_shape[0]))
        return        

    def write_data(self, ch, slice_3D = None):
        """Write a block of data. Only supported for hdf5 datasets.  
        
        :param ch: 3D numpy array to be saved  
        
        :param list slice_3D: list of three python slices e.g. [slice(start,stop,step)]*3 - must match shape of ch  
        
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
        
        :param int axis: axis > 0 is not supported for tiff series  
        
        :param slice s: python slice(start, stop, step) - step must be None for tiff series  
        
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
            
            ch = ch.astype(self.d_type)
            write_tiffseq(ch, SaveDir = self.fname, increment_flag = True,\
                          suffix_len = len(str(self.d_shape[axis])))
        
        return

#### END Class DataFile #######

def read_tiffseq(userfilepath = '', procs = None, s = None):
    """Read a sequence of tiff images from folder.  
    
    :param str userfilepath: path to folder containing images  
    
    :param s: s is either a slice(start, stop, step) or a list of indices to be read  
    
    :type s: slice or list
    
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
    
    S = Parallelize(ImgFileList, imread, procs = procs)

    return S

def write_tiffseq(S, SaveDir = "", increment_flag = False,\
                  suffix_len = None):
    """Write a sequence of tiff images to a directory.  
    
    :param S: numpy array (3D), sequence will be created along axis 0  
    
    :param str SaveDir: path to folder, will create directory if doesn't exist  
    
    :param bool increment_flag: True to write append images to existing ones in folder  
    
    :param int suffix_len: e.g. 4 for 1000 images, 5 for 10,000  
    
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
    
    :param list ListIn: list, each item in the list is a tuple of non-keyworded arguments for f.  
    
    :param func f: function to be parallelized. Signature must not contain any other non-keyworded arguments other than those passed as iterables.  
    
    Example:
    
    .. highlight:: python
    .. code-block:: python
    
        def multiply(x, y, factor = 1.0):
            return factor*x*y
    
        X = np.linspace(0,1,1000)  
        Y = np.linspace(1,2,1000)  
        XY = [ (x, Y[i]) for i, x in enumerate(X)] # List of tuples  
        Z = Parallelize_MultiIn(XY, multiply, factor = 3.0, procs = 8)  
    
    Create as many positional arguments as required, but remember all must be packed into a list of tuples.
    
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
   
def show_header():
    print("\n" + "#"*60 + "\n")
    print("\tWelcome to CTSegNet: AI-based 3D Segmentation")
    print("\n" + "#"*60 + "\n")
    return

def show_endmessage(_str):
    print("\n" + "#"*60 + "\n")
    print("\t" + _str)
    print("\n" + "#"*60 + "\n")
    return
    

def _istiff(fpath, data_tag):
    """
    Returns True if path provided is to a directory of tiffs, False if .hdf5 file.
    Raises ArgumentTypeError if format is not supported.
    """
    # Understand input data format
    if os.path.isdir(fpath):
        tiff_input = True
    elif fpath.split('.')[-1] in ("hdf5", "h5"):
        tiff_input = False
        if data_tag == "":
            raise ArgumentTypeError("dataset-name required for hdf5")
    else:
        raise ArgumentTypeError("input file type not recognized. must be tiff folder or hdf5 file")
    return tiff_input

def get_cropped_shape(crops, d_shape):
    if crops is not None:
        crop_shape = [0]*len(d_shape)
        for idx, crop in enumerate(crops):
            _size = [0,0]
            if (crop[0] is not None):
                if crop[0] >= 0:
                    _size[0] = min(abs(crop[0]), d_shape[idx])
                elif crop[0] < 0:
                    _size[0] = max(0, d_shape[idx] - abs(crop[0]))
            else:
                _size[0] = 0

            if crop[1] is not None:
                if crop[1] >= 0:
                    _size[1] = min(abs(crop[1]), d_shape[idx])
                elif crop[1] < 0:
                    _size[1] = max(0, d_shape[idx] - abs(crop[1]))
            else:
                _size[1] = d_shape[idx]
            crop_shape[idx] = max(_size[1] - _size[0], 0)
    else:
        crop_shape = d_shape
    return tuple(crop_shape)
        
    
    
if __name__ == "__main__":
    
    mem_thres = 20.0
    _message("Data input / output functions.")



