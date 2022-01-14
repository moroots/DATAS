# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 13:07:45 2022

@author: Magnolia
"""

#%% Packages

import datetime
import numpy as np
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.units as munits
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import scipy.ndimage
import pandas as pd
import pyhdf
import pytz

from pyhdf.SD import SD, SDC
import xarray as xr


from pathlib import Path

#%% Function Space

def importing_ceilometer(FilePaths, **kwargs):
    data = {} # all data will be stored into a nested dictionary
    files = {}
    FilePaths = [Path(filePath) for filePath in FilePaths] # converting to Pathlib objects

    for filePath in FilePaths:
        if filePath.is_file() is False:
            print('PATH IS NOT FOUND ON MACHINE')
            return
        
        fileName = filePath.name
        data[fileName] = {} # Nested Dictionary
        with xr.open_dataset(filePath) as file: # importing data as a pyhdf obj
            data[fileName]["time"] = file.time.values
            
            data[fileName]["time"] = file.time.values
            data[fileName]["dateNum"] = [mdates.date2num(t) for t in data[fileName]["time"]]
            
            data[fileName]["range"] = file.range.values
            
            data[fileName]["beta_raw"] = file.beta_raw.values
            data[fileName]["beta_raw"][data[fileName]["beta_raw"] < 0] = 0
            
            data[fileName]["instrument_pbl"] = file.pbl.values
            
            data[fileName]["lat_lon_alt"] = [file.longitude.values, file.latitude.values, file.altitude.values]
            
            if "vars" in kwargs.keys():
                for var in kwargs["vars"]:
                    data[fileName][var] = file[var].values
            
            data[fileName]["datasets"] = list(file.keys())
            files[fileName] = file
            
    return data, files

#%% Execution

if __name__ == '__main__':

    FilePaths = [r"C:/Users/meroo/OneDrive - UMBC/Research/Data/Celiometer/test_data/20210518_Catonsville-MD_CHM160112_000.nc"]
    data, files = importing_ceilometer(FilePaths)

#%% Testbed