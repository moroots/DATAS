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

# Function Space
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
            data[fileName]["beta_raw"][data[fileName]["beta_raw"] == 0] = np.nan
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
         clims=[10**4, 10**6],
         cticks=np.arange(10**4, 10**6, (10**6 - 10**4) / 5),
         xlabel="Datetime (UTC)", cmap="jet",
         **kwargs):

    fig, ax = plt.subplots(figsize=(15, 8))

    for key in data.keys():
        X, Y, Z = (data[key]["dateNum"], data[key]["range"].flatten()/1000, np.abs(data[key]["beta_raw"]))
        im = ax.pcolormesh(X, Y, Z, cmap=cmap, shading="nearest", norm=LogNorm(vmin=clims[0], vmax=clims[1]))

    cbar = fig.colorbar(im, ax=ax, pad=0.01, ticks=cticks)
    cbar.set_label(label=r"Aerosol Backscatter ($Log_{10}$)", size=16)

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
    
    if "target" in kwargs.keys():
        #add rectangle to plot
        ax.add_patch(Rectangle(*kwargs["target"],
             edgecolor = 'pink',
             fill=False,
             lw=1))
        
    if "savefig" in kwargs.keys():
        plt.savefig(f"{kwargs['savefig']}", dpi=300)

    plt.show()

    return (X, Y, Z)


#%% Execution

if __name__ == '__main__':

    figPath = r"C:\Users\meroo\OneDrive - UMBC\Research\Analysis\May2021\Figures"

    FilePaths = [r"C:/Users/meroo/OneDrive - UMBC/Research/Analysis/May2021/data/Ceilometer/GSFC/20210519_TROPOZ_CHM200123_000.nc",
r"C:/Users/meroo/OneDrive - UMBC/Research/Analysis/May2021/data/Ceilometer/GSFC/20210520_TROPOZ_CHM200123_000.nc",
r"C:/Users/meroo/OneDrive - UMBC/Research/Analysis/May2021/data/Ceilometer/GSFC/20210521_TROPOZ_CHM200123_000.nc"]

    data, files = importing_ceilometer(FilePaths, LT=-4)

    parms = {"data": data,
             "ylims": [0, 3],
             "clims": [10**4.5, 10**7],
             "cticks": [10**i for i in range(2, 8)],
             "xlims": ["2021-05-19 20:00", "2021-05-21 00:00"],
             "yticks":np.arange(0.5, 3.1, 0.5),
             "title": r"TROPOZ Ceilometer Lufft CHM15K",
             "cmap": "nipy_spectral",
             "xlabel": "Local Time (UTC -4)",
             "savefig": f"{figPath}\\GSFC_Ceilometer_2021051920_2021052100.png"}

    plot(**parms)
#%% Testbed

path = r"C:\Users\meroo\OneDrive - UMBC\Research\Analysis\DATAS\datas\tutorials\data"