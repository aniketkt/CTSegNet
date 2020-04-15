======
Usage
======

Command-line interface
======================

In addition to the API, CTSegNet provides a command-line interface to perform segmentation and training of new models.Data formats supported are .tiff sequence and hdf5. Example config files are provided in cfg_files/.  

**TRAIN/TEST:** Extract training data from arbitrarily sized CT data and ground-truth pairs::

    python bin/make_training_dataset.py -c cfg_files/setup_train.cfg

Build and train several Unet-like fCNN architectures for an input image size of your choice::

    python bin/train_fCNN.py -t cfg_files/train.cfg -m cfg_files/models/Unet242.cfg

**SEGMENT:** An end-to-end 3D segmentation workflow that binarizes 2D images extracted from 3D CT data using the fCNN model, then rebuilds the corresponding 3D segmentation map. The hdf5 version is optimized for low RAM usage in very large (>50 GB) datasets.::

    python bin/run_segmenter.py -c cfg_files/setup_seg.cfg

**USE HDF5 FORMAT:** Re-package your CT data into hdf5 format, with methods to determine optimal chunk size. Although optional, using hdf5 format accelerates read/write time while slicing through your datasets. Set -c as chunk size in MB or chunk shape z,y,x.::

    python bin/rw_utils/convert_to_hdf5.py -f my_tiff_folder -o output_file.hdf5 -c 20.0
