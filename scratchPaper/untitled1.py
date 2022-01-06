 # -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 12:58:40 2020

@author: Magnolia
"""

import re
import os
import datetime
import numpy as np
import xarray as xr
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.units as munits
from matplotlib.colors import LogNorm


def read_nc(dir_path, filetype='nc'):

    if os.path.isdir(dir_path) is False:
        print('path is not found on machine')
        return

    uL = {}; print(f'Importing lidar data from {dir_path} \n')

    for filename in glob(os.path.join(dir_path, f'*.{filetype}')):
        uL[filename.split('\\')[-1]] = xr.open_dataset(filename)
        print(filename.split('\\')[-1], "-> Imported")

    return uL, list(uL.keys())

# For sorting the raw data
def variables(data, file):
    t = data[file].time

    alt = data[file].range

    # Gathering the backscatter (i.e. beta_raw)
    r = data[file].beta_raw

    # Flip the image
    r = np.array(r).T

    # Filter for negative values
    # r[r < 0] = np.nan
    np.warnings.filterwarnings('ignore')

    ''' Merge Days (couple days) '''

    alt1 = np.min(alt)/1000
    alt2 = np.max(alt)/1000

    x_lims = mdates.date2num(t)
    return x_lims, alt, r, alt1, alt2

def plotting(data, file, lim="auto", title=None, sav_path=None, yrange=[5,0.1], cscale=[10**3.5, 10**8.5], save=1):
    if type(sav_path) is not str:
        sav_path = r'C:\Users\Magnolia\OneDrive - UMBC\Research\Figures\Preliminary'

    # Date Axis Handling #
    t = []
    alt = []
    r = []
    alt1 = []
    alt2 = []

    t, alt, r, alt1, alt2 = variables(data, file)

    converter = mdates.ConciseDateConverter()
    munits.registry[datetime.datetime] = converter

    if type(title) is str: title = title
    else: title = f'{file}'
    fig, ax = plt.subplots(figsize=(12, 3), constrained_layout=True)
    im = plt.imshow(r, extent = [t[0], t[-1], np.mean(alt2), np.mean(alt1)],
                    cmap='jet', aspect='auto')
    cbar = plt.colorbar(extend='both')
    # im.set_clim(vmax=10**-4.5, vmin=10**-8.5)
    cbar.set_label('Aerosol Backscatter')
    ax.set_xlabel('Datetime (UTC)')
    ax.set_ylabel('Altitude (km)')
    ax.set_ylim(yrange)
    ax.set_title(f'{title}')

    plt.gca().invert_yaxis()    # Flip the image so its not upside down

    ax.xaxis_date()
    if type(lim) is "array":
        lims = [np.datetime64(lim[0]), np.datetime64(lim[1])]
        ax.set_xlim(lims)

    fig.autofmt_xdate()

    characters_to_remove = "!()@:"

    pattern = "[" + characters_to_remove + "]"

    new_string = re.sub(pattern, "", title)

    if save == 1: plt.savefig(f"{sav_path}\{new_string}.png", dpi = 600)

    plt.show()

    output = {'t':t, 'alt': alt, 'r':r , 'alt1':alt1, 'alt2':alt}
    return output

# Importing #
path = r"C:\Users\Magnolia\OneDrive - UMBC\Research\Data\Ruben(GOAT)"
# path = r'D:\\'
data1, files = read_nc(path)

#%% Plotting #
# lims = ['2020-09-16 00:00', '2020-09-17 23:00']
files = list(data1.keys())
# data = lidar.plotting(files, title=f'UMBC Celiometer: {lims[0]} to {lims[1]}', lim=lims)
data = plotting(data1, files[4], "auto")

#%%

fig, ax = plt.subplots()
x_lims, alt, r, alt1, alt2 = variables(data1, files[1])
ax.imshow(r, extent = [t[0], t[-1], np.mean(alt2), np.mean(alt1)],
                    cmap='jet', aspect='auto')
