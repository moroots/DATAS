# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 11:31:58 2022

@author: Magnolia
"""

import numpy as np
import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
import h5py
import xarray as xr
import pytz
import matplotlib.units as munits
import matplotlib.dates as mdates
import pandas as pd
import pyhdf 
# def import_data(folderPath, fileName, **kwargs):

#%% Testbed

# def retrieve(filename, print_vars=False):
#     # print(f"Retrieving data file -> {filename} \n")
#     f = h5py.File(filename, 'r')
#     # print(f"DATASETS -> {list(f.keys())} \n")

#     if print_vars:
#         idx,sds = q_datasets(f["DATA"])

#     return f["DATA"]

testFile = r"C:/Users/meroo/OneDrive - UMBC/Research/Analysis/May2021/data/TROPOZ/lidar/groundbased_lidar.o3_nasa.gsfc003_hires_goddard.space.flight.center.md_20210518t000000z_20210519t000000z_001.hdf"


from pyhdf.SD import SD, SDC

file = SD(testFile, SDC.READ)