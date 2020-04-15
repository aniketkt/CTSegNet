=======
Install
=======

ct_segnet with CLI
-------------------

To download the entire package with executables, sample model files and config files, clone the master branch and install locally. To download the .h5 model files when cloning, you will need `Git LFS <https://github.com/git-lfs/git-lfs/wiki/Installation>`_ installed.

::

    git clone https://github.com/aniketkt/CTSegNet.git
    pip install CTSegNet/. 

ct_segnet only
--------------

To install only ct_segnet modules into your python 3 environment, use pip. For compatibility with tensorflow 1.14, please install ct_segnet 1.16 from the tf-1 branch.  

::

    $ pip install git+https://github.com/aniketkt/CTSegNet.git@master#egg=ct_segnet  

in a prepared virtualenv or as root for system-wide installation.

