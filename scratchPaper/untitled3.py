# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 17:49:12 2020

@author: Magnolia

Curtain Plotting
"""

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import xarray as xr
import os
import matplotlib as mpl
import matplotlib.units as munits
import datetime

def read_nc(dir_path, file='all'):

    if file == 'all': file='*.nc'

    try:
        uL = {}; print(f'Importing lidar data from {dir_path} \n')

        for filename in glob(os.path.join(dir_path, file)):
            uL[filename.split('\\')[-1]] = xr.open_dataset(filename)
            print(filename.split('\\')[-1], "-> Imported")

    except:
        print("Oh no, There is an issue")

    else:
        if os.path.isdir(dir_path) is False:
            print('Specified path is not found on machine')

        elif os.path.isfile(os.path.join(dir_path, file)) is False:
            if file != 'all': print('File not found')

    return uL, list(uL.keys())

# For sorting the raw data
def getvar(data, file):
    t = data[file].time

    alt = data[file].range
    alt /=1000

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
    return x_lims, alt, r, alt1, alt2, t


def moving_avg_filtering(r, flt_len=0):

    if flt_len == 0: filter_length = round(len(r)/1000)
    else:
        try: filter_length = flt_len
        except: print('oops')

    mov_avg = []
    for i in range(0, len(r)):
        moving_average = np.convolve(r[i, :], np.ones((filter_length)), mode='same')
        moving_average /= filter_length
        mov_avg.append(moving_average)
    return mov_avg

def median_filter(r, flt_len=25):
    from scipy.signal import medfilt
    filter_length = flt_len
    filter_out = []
    for i in range(0, len(r)):
        median_filter = medfilt(r[i, :], filter_length)
        filter_out.append(median_filter)
    return filter_out

def pcolormesh(alt, r, t, title='default', shade='flat'):

    fig, ax = plt.subplots(figsize=(8, 2))
    im = ax.pcolormesh(t, alt, r, cmap='jet', shading=shade, norm=mpl.colors.LogNorm(vmin=5e3, vmax=1e6))
    plt.colorbar(im, ax=ax)
    ax.set_ylim(0, 5)

    converter = mdates.ConciseDateConverter()
    munits.registry[datetime.datetime] = converter

    fig.autofmt_xdate()

    if title == 'default':
        ax.set_title('pcolormesh curtain')
    else:
        try: ax.set_title(title)
        except: print('Unsupported figure title')

    return

#%% Testbed

path = r"C:\Users\Magnolia\OneDrive - UMBC\Research\Data\Ruben(GOAT)"
file = r'20180701_Catonsville_MD_CHM160112_000.nc'
data1, names = read_nc(path, file=file)

xlims, alt, r, alt1, alt2, t = getvar(data1, file)

#%% plotting
pcolormesh(alt, r, t, title='pcolormesh with Flat Shading')

pcolormesh(alt, r, t, title='pcolormesh with Gouraud Shading', shade='gouraud')

test = moving_avg_filtering(r, 3)
pcolormesh(alt, test, t, title='Pcolormesh with Moving Average Filter', shade='gouraud')

test2 = median_filter(r, 7)
pcolormesh(alt, test2, t, title='Pcolormesh with Median Average Filter', shade='gouraud')

