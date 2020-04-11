# CTSegNet

CTSegNet is a package for 3D Segmentation of X-ray Tomographic Data using 2D fully Convolutional Neural Networks (fCNN). With this package you can:

1. Build and train Unet-like fCNN architectures for an input size of your choice.
2. Extract training / testing data from arbitrarily sized 3D datasets (.hdf5 or tiff sequences are supported)
3. Segment 3D datasets with an end-to-end algorithm that extracts 2D patches from 3D data, predicts segmented images and rebuilds the corresponding 3D segmentation map.
4. View orthogonal slices of input CT data overlain with segmentation maps. Viewing utilities are built on matplotlib

Some unique features include:
 - Rapid, memory efficient slicing so you can visualize and segment 100GB+ datasets from your workstation.
 - A patching algorithm that allows you to segment arbitrarily sized data with a 2D CNN of any input size.
 - Select only parts of your data for segmentation or test models on slices of your data, with a few lines of code.

This is version 2.0 that requires tensorflow 2.0. For tensorflow 1.14, please install ct_segnet 1.2 from the tf-1 branch


Notes for installation. Use pip install with the setup.py file. You will also need ImageStackPy package, which is available at github.com/aniketkt/ImageStackPy.




