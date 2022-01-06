# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 11:13:44 2020

@author: mroots

This script is designed to plot lidar curtains from netcdf files

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

class lidar:
    # For loading the raw data
    def read_nc(dir_path, filetype='nc'):

        uL = {}

        if os.path.isdir(dir_path) is False:
            print('path is not found on machine')
            return

        print(f'Importing lidar data from {dir_path} \n')

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
        r[r < 0] = np.nan
        np.warnings.filterwarnings('ignore')

        ''' Merge Days (couple days) '''

        alt1 = np.min(alt)/1000
        alt2 = np.max(alt)/1000

        x_lims = mdates.date2num(t)
        return x_lims, alt, r, alt1, alt2

    def plotting(data, lim, title=None, sav_path=None, yrange=[5,0.1], cscale=[10**3.5, 10**8.5], save=1):

        files = list(data.keys())

        if type(sav_path) is not str:
            sav_path = r'C:\Users\Magnolia\OneDrive - UMBC\Research\Figures\Preliminary'

        i = 0
        t = []
        alt = []
        r = []
        alt1 = []
        alt2 = []

        for file in files:
            uL_t, uL_alt, uL_r, uL_alt1, uL_alt2 = lidar.variables(data, file)

            if i == 0:
                t, alt, r, alt1, alt2 = lidar.variables(data, file)
            i += 1
            if i > 1:
                t = np.append(t,uL_t)
                alt = np.append(alt, uL_alt)
                r = np.hstack((r, uL_r))
                alt1 = np.append(alt1, uL_alt1)
                alt2 = np.append(alt2, uL_alt2)

        # Date Axis Handling #
        converter = mdates.ConciseDateConverter()
        munits.registry[datetime.datetime] = converter

        if type(title) is str: title = title
        else: title = 'blank'
        fig, ax = plt.subplots(figsize=(12, 3), constrained_layout=True)
        im = plt.imshow(r, extent = [t[0], t[-1], np.mean(alt2), np.mean(alt1)],
                        cmap='jet', aspect='auto', norm=LogNorm())
        cbar = plt.colorbar(extend='both')
        im.set_clim(vmin=10**3.5, vmax=10**8.5)
        cbar.set_label('Aerosol Backscatter')
        ax.set_xlabel('Datetime (UTC)')
        ax.set_ylabel('Altitude (m)')
        ax.set_ylim(yrange)
        ax.set_title(f'{title}')

        plt.gca().invert_yaxis()    # Flip the image so its not upside down



        ax.xaxis_date()

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


#%% For Troubleshooting

# Importing #
path = r"C:\Users\Magnolia\OneDrive - UMBC\Research\Data\Ruben(GOAT)"
# path = r'D:\\'
data1, files = lidar.read_nc(path)

# Plotting #
lims = ['2018-07-01 00:00', '2018-07-02 23:00']
# data = lidar.plotting(files, title=f'UMBC Celiometer: {lims[0]} to {lims[1]}', lim=lims)
data = lidar.plotting(data1, lims)

# datetime.datetime.strptime(str(test), "%Y%m%d.%H") + datetime.timedelta(hours = time[0])