# -*- coding: utf-8 -*-
"""
in progress - this will be coupled with tomostream code.  

"""
import os
import numpy as np
import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
import h5py
import sys
from tensorflow import keras
from IPython.display import clear_output
import random
import pandas as pd
from multiprocessing import cpu_count
from ct_segnet.data_utils.data_io import Parallelize
from ct_segnet.stats import *#ROC, calc_jac_acc, calc_dice_coeff, fidelity
from ct_segnet.data_utils.data_augmenter import run_augmenter



def Trainer():
    
    def __init__(self, model_filename = None, \
                 model = None, \
                 model_name = "unknown", \
                 weight_file_name = None, \
                 GPU_mem_limit = 16.0):
        
        raise NotImplementedError("coming soon")
        return
        
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
          try:
            tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=GPU_mem_limit*1000.0)])
          except RuntimeError as e:
            print(e)        
        
        # use model directly if provided
        if model is not None:
            self.model = model
            self.model_name = model_name
        else: # load from a file
            self.model_name = os.path.split(model_filename)[-1].split('.')[0]
            self.model = load_model(model_filename, custom_objects = custom_objects_dict)

    def get_random_slice_coordinates(self, vol_shape, prev_list = None):
        
        random_axis = np.random.randint(3)
        random_idx = np.random.randint(vol_shape[random_axis])
        
        # future to-do: sample from a cropped volume; efficient sampling (e.g. without replacement, based on model accuracy) using a prev_list data object
        # possible prev_list is a df with columns ["idx", "axis", "IoU-prev", "SNR-est"]
        
        return random_axis, random_idx
    
    def extract_from_slice(self, img, n_patches = (2,2), overlap = 20):
        
        return p # list of patches ( = model size) extracted from an image
        
#         patch_shape = self.model.output_shape[1::-1]
#         patch_size, patch_size = patch_shape
        
#         if type(n_patches) is not tuple:
#             n_patches = (n_patches, n_patches)
            
#         p = np.asarray([cv2.resize(p, (n_patches[1]*patch_size[1], n_patches[0]*patch_size[0])) for ii in range(p.shape[0])])

        
    def extract_from_volume(self, vol, n_patches = (2,2,2), overlap = 20, crops = None):
        
        return p # list of patches ( = model size) extracted from a volume

















