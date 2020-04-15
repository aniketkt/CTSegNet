========
CTSegNet
========


.. image:: source/img/project-logo.png
   :width: 640px
   :alt: project


**CTSegNet** is a package for end-to-end 3D segmentation workflow for large X-ray tomographic datasets using 2D fully convolutional neural networks (fCNN).


Features
--------

* Command-line interface for deploying 3D segmentation workflow using 2D fCNNs as well training your own Unet-like models with data-augmentation.
* Read/write utilities support tiff series and hdf5 format. The hdf5 implementation exploits chunking to minimize RAM usage in very large (>50 GB) datasets.
* Write your own scripts to test models and visualize segmentation results using the API.

Contribute
----------

* Documentation: https://github.com/aniketkt/CTSegNet/tree/master/doc
* Issue Tracker: https://github.com/aniketkt/CTSegNet/docs/issues
* Source Code: https://github.com/aniketkt/CTSegNet/

Content
-------

.. toctree::
   :maxdepth: 1

   source/about
   source/install
   source/usage
   source/api
   source/credits
