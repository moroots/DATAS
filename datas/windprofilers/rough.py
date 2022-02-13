# -*- coding: utf-8 -*-
"""
Created on Sun Jan  9 21:16:57 2022

@author: meroo
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.units as munits
import pandas as pd
import datetime
import scipy.io



filePath = r"C:\Users\meroo\OneDrive - UMBC\Research\Analysis\May2021\data\RWP"
data = {}
fileNames = [f"data_{j}.mat" for j in range(1,8)]
for fileName in fileNames:
    data[fileName] = scipy.io.loadmat(f"{filePath}\{fileName}")
    data[fileName]["timestamp"] = [datetime.datetime.fromordinal(int(matlab_datenum)) + datetime.timedelta(days=matlab_datenum%1) - datetime.timedelta(days = 366, hours=4) for matlab_datenum in data[fileName]["TD"].flatten()]
    data[fileName]["datenum"] = mdates.date2num(data[fileName]["timestamp"])
    
#%% 

import matplotlib.pyplot as plt
import matplotlib as mpl



def wind_dir_colormap(cmap_name):
    custom_maps = {"cardinal8":{"cmap":(mpl.colors.ListedColormap(['red', 'darkorange', "gold", "forestgreen", "turquoise", "mediumblue", "darkviolet", "deeppink"])), "bounds": [0, 45, 90, 135, 180, 225, 270, 315, 360]}}
    
    ncmap = custom_maps[cmap_name]["cmap"]
    bounds = custom_maps[cmap_name]["bounds"]
    ncmap.set_under([1,1,1])
    ncmap.set_over([0,0,0])
    
    nnorm = mpl.colors.BoundaryNorm(bounds, ncmap.N)
    return ncmap, nnorm

def RWP_plot(data, **kwargs):
    
    """ 
    data: dictionary object with needed vars
    """
    
    ncmap, nnorm = wind_dir_colormap("cardinal8")
    
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(15,8))
    
    for fileName in data.keys():
        X, Y, SPD, DIR = (data[fileName]["datenum"],
                          data[fileName]["ALT"].flatten(),
                          data[fileName]["SPD"],
                          data[fileName]["DIR"])
        
        im1 = ax1.contourf(X, Y, SPD, cmap="turbo")
        im2 = ax2.contourf(X, Y, DIR, cmap=ncmap)
   
    cbar1 = fig.colorbar(im1,ax=ax1, ticks=[0, 10, 20, 30, 40, 50], pad=0.01,
                 spacing='proportional', label="Wind Speed (m/s)")
        
    if "degrees" in kwargs.keys() and kwargs["degrees"]:
        cbar2 = fig.colorbar(mpl.cm.ScalarMappable(cmap=ncmap, norm=nnorm), ax=ax2, ticks=[0, 90, 180, 270, 360], pad=0.01, spacing='proportional', label="Wind Direction (degrees)")
    
    else: 
        cbar2 = fig.colorbar(mpl.cm.ScalarMappable(cmap=ncmap, norm=nnorm), ax=ax2,ticks=[22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5], pad=0.01,
                  spacing='proportional', label="Wind Direction")
        cbar2.ax.set_yticklabels(['NNE', 'ENE', 'ESE', 'SSE', 'SSW', 'WSW', 'WNW', 'NNW'])
        
    if "title" in kwargs.keys():
            plt.suptitle(kwargs["title"], fontsize=18)
    else: plt.suptitle(r"Radio Wind Profiler (RWP)", fontsize=18, x=0.45, y=0.92)
    
    ax1.set_ylabel("Altitude (km AGL)", fontsize=16)
    ax2.set_ylabel("Altitude (km AGL)", fontsize=16)
    ax2.set_xlabel("Local Time (UTC-4)", fontsize=16)
    
    if "xlims" in kwargs.keys():
        lim = kwargs["xlims"]
        lims = [np.datetime64(lim[0]), np.datetime64(lim[-1])]
        ax1.set_xlim(lims)
    
    if "ylims" in kwargs.keys():
        ax1.set_ylim(kwargs["ylims"][0:2])
        ax1.set_yticks(np.arange(kwargs["ylims"][0], kwargs["ylims"][1], 
                                step=kwargs["ylims"][2]))
    
    if "surface" in kwargs.keys():
        df = kwargs["surface"][0]
        dummy = np.ones(len(df))*kwargs["surface"][1]
        # ax.scatter(df.index, dummy, c=df, cmap=ncmap, norm=nnorm)
    
    converter = mdates.ConciseDateConverter()
    munits.registry[datetime.datetime] = converter
    
    ax1.xaxis_date()
    ax2.xaxis_date()
    
    if "savefig" in kwargs.keys():
        plt.savefig(f"{kwargs['savefig']}", dpi=600)
    
    plt.show()
    plt.rcParams.update(plt.rcParamsDefault)
    
    return

parms = {""}

RWP_plot(data, savefig=r"C:\Users\meroo\OneDrive - UMBC\Research\Analysis\May2021\Figures\RWP_20210517_20210523_degrees.png", degrees=True)

RWP_plot(data, savefig=r"C:\Users\meroo\OneDrive - UMBC\Research\Analysis\May2021\Figures\RWP_20210517_20210523.png", degrees=False)


#%% Testbed


# fig, ax = plt.subplots(figsize=(6, 1))
# fig.subplots_adjust(bottom=0.5)

# # cmap = (mpl.colors.ListedColormap(['red', 'goldenrod', "yellowgreen", "green", "turquoise", "dodgerblue", "blue", "slateblue", "rebeccapurple", "fuchsia", "orchid", "pink", "red"]))

# cmap = (mpl.colors.ListedColormap(['red', 'darkorange', "gold", "forestgreen", "turquoise", "mediumblue", "darkviolet", "deeppink"]))

# bounds = [0, 45, 90, 135, 180, 225, 270, 315, 360]
# norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
# fig.colorbar(
#     mpl.cm.ScalarMappable(cmap=cmap, norm=norm),
#     cax=ax,
#     ticks=[0, 90, 180, 270, 360],
#     spacing='proportional',
#     orientation='horizontal',
#     label='Discrete intervals, some other units',
# )