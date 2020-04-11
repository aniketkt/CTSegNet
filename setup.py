#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 10:56:50 2018
@author: atekawade
"""

from setuptools import setup, find_packages

setup(
    # Needed to silence warnings (and to be a worthwhile package)
    name='ct_segnet',
    url='https://github.com/aniketkt/CTSegNet',
    author='Aniket Tekawade',
    author_email='atekawade@anl.gov',
    # Needed to actually package something
    packages= ['ct_segnet', 'ct_segnet.data_utils', 'ct_segnet.model_utils'],
    # Needed for dependencies
    install_requires=['tensorflow-gpu==2.0',
			'numpy', 'scikit-image', 'matplotlib', 'opencv-python', 'pandas', 'tifffile', 'scipy','scikit-learn','ConfigArgParse', 'tqdm'],
    version='1.2',
    license='BSD',
    description='Automated 3D segmentation powered by 2D convolutional neural networks',
    # We will also need a readme eventually (there will be a warning)
    long_description=open('README.md').read(),
)


