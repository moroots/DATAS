# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 13:07:45 2022

@author: Maurice Roots

"""

#%% Packages

# Utilities
from pathlib import Path
from datetime import datetime

# Data Processing
import numpy as np
import xarray as xr

# Plotting
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.units as munits
from matplotlib.colors import LogNorm
#%% Function Space

def importing_ceilometer(FilePaths, variables=None, LT=None, **kwargs):
    data = {} # all data will be stored into a nested dictionary
    files = {}
    FilePaths = [Path(filePath) for filePath in FilePaths] # converting to Pathlib objects

    for filePath in FilePaths:
        if filePath.is_file() is False:
            print('PATH IS NOT FOUND ON MACHINE')
            return

        fileName = filePath.name
        data[fileName] = {} # Nested Dictionary
        with xr.open_dataset(filePath) as file: # importing data as a xarrays instance
            data[fileName]["datetime"] = file.time.values
            data[fileName]["dateNum"] = np.array([mdates.date2num(t) for t in data[fileName]["datetime"]])
            if LT: data[fileName]["dateNum"] = data[fileName]["dateNum"] + (LT/24)
            data[fileName]["range"] = file.range.values
            data[fileName]["beta_raw"] = file.beta_raw.values
            data[fileName]["beta_raw"][data[fileName]["beta_raw"] == 0] = np.float64(np.nan)
            data[fileName]["beta_raw"] = data[fileName]["beta_raw"].T
            data[fileName]["instrument_pbl"] = file.pbl.values
            data[fileName]["lat_lon_alt"] = [file.longitude.values, file.latitude.values, file.altitude.values]

            if "vars" in kwargs.keys():
                for var in kwargs["vars"]:
                    data[fileName][var] = file[var].values

            data[fileName]["datasets"] = list(file.keys())
            files[fileName] = file

    return data, files


def plot(data, 
         clims=[3.5, 8.5], 
         cticks=np.arange(3.5, 8.6, 0.5), 
         xlabel="Datetime (UTC)",
         **kwargs):

    fig, ax = plt.subplots(figsize=(15, 8))

    for key in data.keys():
        X, Y, Z = (data[key]["dateNum"], data[key]["range"].flatten()/1000, np.log10(np.abs(data[key]["beta_raw"])))
        # X, Y, Z = (data[key]["dateNum"], data[key]["range"].flatten()/1000, np.abs(data[key]["beta_raw"]))
        im = ax.pcolormesh(X, Y, Z, cmap='jet', shading="nearest", vmin=clims[0], vmax=clims[1])

    cbar = fig.colorbar(im, ax=ax, pad=0.01, ticks=cticks)
    cbar.set_label(label=r"Aerosol Backscatter ($Log_{10}$)", size=16, weight="bold")

    if "title" in kwargs.keys():
        plt.title(kwargs["title"], fontsize=20)
    else: plt.title(r"Ceilometer Backscatter", fontsize=20)

    ax.set_ylabel("Altitude (km AGL)", fontsize=18)
    ax.set_xlabel(xlabel, fontsize=18)

    if "xlims" in kwargs.keys():
        lim = kwargs["xlims"]
        lims = [np.datetime64(lim[0]), np.datetime64(lim[-1])]
        ax.set_xlim(lims)

    if "ylims" in kwargs.keys():
        ax.set_ylim(kwargs["ylims"])

    if "yticks" in kwargs.keys():
        ax.set_yticks(kwargs["yticks"], fontsize=20)
        
    plt.setp(ax.get_yticklabels(), fontsize=16)
    plt.setp(ax.get_xticklabels(), fontsize=16)
    cbar.ax.tick_params(labelsize=16)
    
    converter = mdates.ConciseDateConverter()
    munits.registry[datetime] = converter

    ax.xaxis_date()

    if "savefig" in kwargs.keys():
        plt.savefig(f"{kwargs['savefig']}", dpi=300)

    plt.show()

    return


#%% Execution

if __name__ == '__main__':

    figPath = r"C:\Users\meroo\OneDrive - UMBC\Research\Analysis\May2021\Figures"

    FilePaths = [r"C:/Users/meroo/OneDrive - UMBC/Research/Analysis/DATAS/datas/lidar/samples/20200308_Catonsville-MD_CHM160112_000.nc"]

    data, files = importing_ceilometer(FilePaths)

    parms = {"data": data,
             "ylims": [0, 5],
             "clims": [4, 6],
             "yticks":np.arange(0.5, 5.1, 0.5),
             "title": r"UMBC Lufft CHM15K",
             "savefig": f"{figPath}\\UMBC_Ceilometer_20200508.png"}

    plot(**parms)
#%% Testbed