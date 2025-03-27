#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ## Import necessary libraries

# Uncomment and run the following lines to authenticate with Earth Engine
# using the notebook or gcloud mode:
# earthengine authenticate --auth_mode=notebook --force
# earthengine authenticate --auth_mode=gcloud --quiet

from termcolor import colored  # Utility for colored terminal text output
print(colored('Successfully imported termcolor.colored', 'green'))

import ee  # Google Earth Engine Python API
print(colored('Successfully imported ee', 'green'))

from IPython import display  # Display utilities for Jupyter notebooks
print(colored('Successfully imported IPython display', 'green'))

import math  # Mathematical functions
print(colored('Successfully imported math', 'green'))

from matplotlib import pyplot as plt  # Visualization library for plots and images
print(colored('Successfully imported matplotlib.pyplot', 'green'))

import numpy as np  # Library for numerical operations on arrays and matrices
print(colored('Successfully imported numpy', 'green'))

import pandas as pd  # Data analysis and manipulation library
print(colored('Successfully imported pandas', 'green'))

import seaborn as sns  # Data visualization library based on matplotlib
print(colored('Successfully imported seaborn', 'green'))

from scipy import ndimage  # Multidimensional image processing
print(colored('Successfully imported scipy.ndimage', 'green'))

import rasterio  # Reading and writing geospatial raster data
print(colored('Successfully imported rasterio', 'green'))

from rasterio.mask import mask  # Function to mask raster data with shapes
print(colored('Successfully imported rasterio.mask', 'green'))

import tempfile  # Temporary file creation utilities
print(colored('Successfully imported tempfile', 'green'))

import urllib.request as urllib  # For downloading files from the internet
print(colored('Successfully imported urllib.request', 'green'))

import zipfile  # Tools to handle ZIP archives
print(colored('Successfully imported zipfile', 'green'))

import datetime  # Handling date and time
print(colored('Successfully imported datetime', 'green'))

import time  # Time-related functions
print(colored('Successfully imported time', 'green'))

import os  # OS module for interacting with the operating system
print(colored('Successfully imported os', 'green'))

import string  # String handling utilities
print(colored('Successfully imported string', 'green'))

import glob  # File pattern matching
print(colored('Successfully imported glob', 'green'))

from osgeo import gdal  # Geospatial Data Abstraction Library for raster data processing
print(colored('Successfully imported osgeo.gdal', 'green'))

from shapely.geometry import shape, mapping, box  # Geometric objects and operations
print(colored('Successfully imported shapely.geometry', 'green'))

from shapely.ops import transform  # Geometric transformations
print(colored('Successfully imported shapely.ops', 'green'))

import pyproj  # Coordinate reference system transformations
print(colored('Successfully imported pyproj', 'green'))

# import tensorflow as tf  # TensorFlow for building neural networks (version 2.x)
import tensorflow.compat.v1 as tf  # TensorFlow compatibility mode for version 1.x
tf.disable_v2_behavior()  # Disable TensorFlow 2.x behaviors and enable 1.x style
print(colored('Successfully imported tensorflow.compat.v1', 'green'))

# Uncomment the following line to authenticate with Google Earth Engine
# ee.Authenticate()

ee.Initialize(project='workspace-ipam')
print(colored('Google Earth Engine API initialized successfully', 'green'))
