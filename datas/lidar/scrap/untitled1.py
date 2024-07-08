# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 12:04:28 2023

@author: meroo
"""

# Utilities
from pathlib import Path
from datetime import datetime

# Data Processing
import numpy as np
import netCDF4 as nc

# # Function Space
# def importing_ceilometer(FilePaths, variables=None, LT=None, **kwargs):
#     data = {} # all data will be stored into a nested dictionary
#     files = {}
#     FilePaths = [Path(filePath) for filePath in FilePaths] # converting to Pathlib objects

#     for filePath in FilePaths:
#         if filePath.is_file() is False:
#             print('PATH IS NOT FOUND ON MACHINE')
#             return

#         fileName = filePath.name
#         data[fileName] = {} # Nested Dictionary
#         with xr.open_dataset(filePath) as file: # importing data as a xarrays instance
#             data[fileName]["datetime"] = file.time.values
#             data[fileName]["dateNum"] = np.array([mdates.date2num(t) for t in data[fileName]["datetime"]])
#             if LT: data[fileName]["dateNum"] = data[fileName]["dateNum"] + (LT/24)
#             data[fileName]["range"] = file.range.values
#             data[fileName]["beta_raw"] = file.beta_raw.values
#             data[fileName]["beta_raw"][data[fileName]["beta_raw"] == 0] = np.nan
#             data[fileName]["beta_raw"] = data[fileName]["beta_raw"].T
#             data[fileName]["instrument_pbl"] = file.pbl.values
#             data[fileName]["lat_lon_alt"] = [file.longitude.values, file.latitude.values, file.altitude.values]

#             if "vars" in kwargs.keys():
#                 for var in kwargs["vars"]:
#                     data[fileName][var] = file[var].values

#             data[fileName]["datasets"] = list(file.keys())
#             files[fileName] = file

#     return data, files


def unpack(filePath):
    dataset = {}
    
    with nc.Dataset(filePath) as file: # importing data as a xarrays instance
        for key in file.variables.keys():
            dataset[key]= file.variables[key][:].data
            
    return dataset

def importing(filePath, file_extension=".nc"):
    
    filePath = Path(filePath)
    
    if filePath.is_dir(): 
        filePaths = filePath.glob(f"*{file_extension}")
        
    elif filePath.is_file():
        filePaths = [filePath]
    
    else:
        print("PATH IS NOT FOUND ON MACHINE")
        return
        
    data = {}
    for file in filePaths:
        data[file.name] = unpack(file)
        
    return data

    
path = r"C:\Users\meroo\OneDrive - UMBC\Research\Analysis\DATAS\datas\tutorials\data\ceilometer"

dataset = importing(path)
