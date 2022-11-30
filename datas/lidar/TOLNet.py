# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 11:31:58 2022

@author: Maurice Roots
"""


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

from netCDF4 import Dataset

#%% Function Space

class TOLNET:
    ''' class to handle TOLNET datasets'''

    def __init__(self, filepaths):
        self.data = self.read_tolnet(fielapths)

    def clean_vars(self, var):
        var[var<=-999] = np.nan
        return var

    def read_tolnet(self, FilePaths, **kwargs):
        data={}

        FileNames = [filePath.split("/")[-1] for filePath in FilePaths]

        for fileName, filePath in zip(FileNames, FilePaths):
            data[fileName] = {}
            file = SD(filePath, SDC.READ)

            data[fileName]["alt"] = file.select('ALTITUDE').get()
            data[fileName]["datetime"] = file.select('DATETIME.START').get() + 10957
            data[fileName]["datetime"] = data[fileName]["datetime"].flatten()
            if "local" in kwargs.keys():
                data[fileName]["datetime"] = data[fileName]["datetime"] + (kwargs["local"]/24)
            data[fileName]["O3MX"] = file.select('O3.MIXING.RATIO.VOLUME_DERIVED').get()
            data[fileName]["O3MX"] = self.clean_vars(data[fileName]["O3MX"])*1000
            data[fileName]["O3ND"] = file.select('O3.NUMBER.DENSITY_ABSORPTION.DIFFERENTIAL').get()
            data[fileName]["O3ND"] = self.clean_vars(data[fileName]["O3ND"])
            data[fileName]["O3ND_Error"] = file.select("O3.NUMBER.DENSITY_ABSORPTION.DIFFERENTIAL_UNCERTAINTY.COMBINED.STANDARD").get()
            data[fileName]["O3ND_Error"] = self.clean_vars(data[fileName]["O3ND_Error"])
            data[fileName]["O3MX_Error"] = file.select("O3.MIXING.RATIO.VOLUME_DERIVED_UNCERTAINTY.COMBINED.STANDARD").get()

            data[fileName]["O3MX_Error"] = self.clean_vars(data[fileName]["O3ND_Error"])

            if "vars" in kwargs.keys():
                for var in kwargs["vars"]:
                    data[fileName][var] = file.select(var).get()

            data[fileName]["datasets"] = file.datasets()

            file.end()
        return data

    def O3_curtain_colors(self):
        ncolors = [np.array([255,  140,  255]) / 255.,
           np.array([221,  111,  242]) / 255.,
           np.array([187,  82,  229]) / 255.,
           np.array([153,  53,  216]) / 255.,
           np.array([119,  24,  203]) / 255.,
           np.array([0,  0,  187]) / 255.,
           np.array([0,  44,  204]) / 255.,
           np.array([0,  88,  221]) / 255.,
           np.array([0,  132,  238]) / 255.,
           np.array([0,  165,  255]) / 255.,
           np.array([0,  235,  255]) / 255.,
           np.array([39,  255,  215]) / 255.,
           np.array([99,  255,  150]) / 255.,
           np.array([163,  255,  91]) / 255.,
           np.array([211,  255,  43]) / 255.,
           np.array([255,  255,  0]) / 255.,
           np.array([250,  200,  0]) / 255.,
           np.array([255,  159,  0]) / 255.,
           np.array([255,  111,  0]) / 255.,
           np.array([255,  63,  0]) / 255.,
           np.array([255,  0,  0]) / 255.,
           np.array([216,  0,  15]) / 255.,
           np.array([178,  0,  31]) / 255.,
           np.array([140,  0,  47]) / 255.,
           np.array([102,  0,  63]) / 255.,
           np.array([200,  200,  200]) / 255.,
           np.array([140,  140,  140]) / 255.,
           np.array([80,  80,  80]) / 255.,
           np.array([52,  52,  52]) / 255.,
           np.array([0,0,0]) ]

        ncmap = mpl.colors.ListedColormap(ncolors)
        ncmap.set_under([1,1,1])
        ncmap.set_over([0,0,0])
        bounds =   [0.001, 10, 20, 30, 40, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 76, 81, 84, 87, 90, 92, 94, 96, 102, 125, 150, 200, 300, 600]
        # bounds = [0.001,  4,  8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 125, 150, 200, 300, 600]
        nnorm = mpl.colors.BoundaryNorm(bounds, ncmap.N)
        return ncmap, nnorm

    def plot(self, data, smooth=True, xlabel="Datetime (UTC)",  **kwargs):


        if 'fontsize' in kwargs.keys():
            fontsize = kwargs["fontsize"]
        else:
            fontsize = 14

        if 'plt' in kwargs.keys():
            fig, ax = kwargs["plt"]
        else:
            fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=1, figsize=(15,14))

        fig = plt.figure(figsize=(15, 8))
        ax = plt.subplot(111)
        ncmap, nnorm = self.O3_curtain_colors()

        plt.rc('font', size=fontsize) #controls default text size
        plt.rc('axes', titlesize=fontsize) #fontsize of the title
        plt.rc('axes', labelsize=fontsize) #fontsize of the x and y labels
        plt.rc('xtick', labelsize=fontsize) #fontsize of the x tick labels
        plt.rc('ytick', labelsize=fontsize) #fontsize of the y tick labels
        plt.rc('legend', fontsize=fontsize) #fontsize of the legend

        for fileName in data.keys():
            # data[fileName]["O3MX_smooth"] = np.zeros(data[fileName]["O3MX"].shape)
            if smooth is True:
            #     data[fileName]["O3MX_smooth"] = scipy.ndimage.zoom(data[fileName]["O3MX"], 3)
                X, Y, Z = (data[fileName]["datetime"],data[fileName]["alt"]/1000, data[fileName]["O3MX"].T)
            im = ax.pcolormesh(X, Y, Z, cmap=ncmap, norm=nnorm, shading="nearest")

        cbar = fig.colorbar(im, ax=ax, pad=0.01, ticks=[0.001, 50, 60, 70, 90, 100, 300])
        cbar.set_label(label=r"Ozone Mixing Ratio (ppbv)", size=16)

        if "title" in kwargs.keys():
            plt.title(kwargs["title"], fontsize=18)
        else: plt.title(r"$O_3$ Mixing Ratio Profile ($ppb_v$)", fontsize=18)

        ax.set_ylabel("Altitude (km AGL)", fontsize=18)
        ax.set_xlabel(xlabel, fontsize=18)

        if "xlims" in kwargs.keys():
            lim = kwargs["xlims"]
            lims = [np.datetime64(lim[0]), np.datetime64(lim[-1])]
            ax.set_xlim(lims)

        if "ylims" in kwargs.keys():
            ax.set_ylim(kwargs["ylims"][0:2])
            ax.set_yticks(np.arange(kwargs["ylims"][0], kwargs["ylims"][1], step=kwargs["ylims"][2]))

        if "surface" in kwargs.keys():
            df = kwargs["surface"][0]
            dummy = np.ones(len(df))*kwargs["surface"][1]
            ax.scatter(df.index, dummy, c=df, cmap=ncmap, norm=nnorm)

        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))

        if "savefig" in kwargs.keys():
            plt.savefig(f"{kwargs['savefig']}", dpi=600)

        plt.rcParams.update(plt.rcParamsDefault)

        return

#%%

if __name__ == "__main__":

    files = [
        r"C:/Users/meroo/OneDrive - UMBC/Research/Analysis/May2021/data/TROPOZ/lidar/groundbased_lidar.o3_nasa.gsfc003_hires_goddard.space.flight.center.md_20210518t000000z_20210519t000000z_001.hdf",
        r"C:/Users/meroo/OneDrive - UMBC/Research/Analysis/May2021/data/TROPOZ/lidar/groundbased_lidar.o3_nasa.gsfc003_hires_goddard.space.flight.center.md_20210519t000000z_20210520t000000z_001.hdf",
        r"C:/Users/meroo/OneDrive - UMBC/Research/Analysis/May2021/data/TROPOZ/lidar/groundbased_lidar.o3_nasa.gsfc003_hires_goddard.space.flight.center.md_20210520t000000z_20210521t000000z_001.hdf"]

    surface_files = [
        r"C:/Users/meroo/OneDrive - UMBC/Research/Analysis/May2021/data/TROPOZ/TOLNet-SurfaceObs_GSFC_20210521_R0.ict",
        r"C:/Users/meroo/OneDrive - UMBC/Research/Analysis/May2021/data/TROPOZ/TOLNet-SurfaceObs_GSFC_20210519_R0.ict",
        r"C:/Users/meroo/OneDrive - UMBC/Research/Analysis/May2021/data/TROPOZ/TOLNet-SurfaceObs_GSFC_20210520_R0.ict"]

    figPath = r"C:\Users\meroo\OneDrive - UMBC\Research\Analysis\May2021\Figures"

    data = TOLNET.read_tolnet(files, local= -4,)

    surface_data = pd.concat((pd.read_csv(f, sep=",", header=0, skiprows=41) for f in surface_files), ignore_index=True)
    #%%
    parms = {"data": data,
             "ylims":[0.1, 3.1, 0.4],
             "title":r"TROPOZ $O_3$ Profile",
             "xlabel": "Local Time (UTC -4)"
             }

    # tolnet_curtains(**parms, xlims=["2021-05-18 12:00", "2021-05-23 00:00"], savefig=f"{figPath}\\TROPOZ_20210518_20210523.png")

    # tolnet_curtains(**parms, xlims=["2021-05-18 12:00", "2021-05-21 00:00"], savefig=f"{figPath}\\TROPOZ_20210518_20210521.png")

    TOLNET.plot(**parms, xlims=["2021-05-19 20:00", "2021-05-21 00:00"], savefig=f"{figPath}\\TROPOZ_2021051920_2021052100.png")

#%% Testbed
    idx = np.where(data["groundbased_lidar.o3_nasa.gsfc003_hires_goddard.space.flight.center.md_20210519t000000z_20210520t000000z_001.hdf"]["alt"] <= 500)[0][-1]

    plt.figure()
    plt.plot(data["groundbased_lidar.o3_nasa.gsfc003_hires_goddard.space.flight.center.md_20210519t000000z_20210520t000000z_001.hdf"]["O3MX"][:, idx], "-*")


#%%

# def tolnet_lineplots(data, xlabel="Datetime (UTC)",  **kwargs):

#     fig = plt.figure(figsize=(15, 8))
#     ax = plt.subplot(111)

#     plt.rc('font', size=16) #controls default text size
#     plt.rc('axes', titlesize=16) #fontsize of the title
#     plt.rc('axes', labelsize=16) #fontsize of the x and y labels
#     plt.rc('xtick', labelsize=16) #fontsize of the x tick labels
#     plt.rc('ytick', labelsize=16) #fontsize of the y tick labels
#     plt.rc('legend', fontsize=16) #fontsize of the legend

#     for fileName in data.keys():
#         idx = np.where(data["groundbased_lidar.o3_nasa.gsfc003_hires_goddard.space.flight.center.md_20210519t000000z_20210520t000000z_001.hdf"]["alt"] <= 500)[0][-1]
#         X, Y = (data[fileName]["datetime"], data[fileName]["O3MX"].T)
#         data[fileName]["alt"]/1000,
#         im = ax.pcolormesh(X, Y, Z, cmap=ncmap, norm=nnorm, shading="nearest")

#     cbar = fig.colorbar(im, ax=ax, pad=0.01, ticks=[0.001, 50, 60, 70, 90, 100, 300])
#     cbar.set_label(label=r"Ozone Mixing Ratio (ppbv)", size=16)

#     if "title" in kwargs.keys():
#         plt.title(kwargs["title"], fontsize=18)
#     else: plt.title(r"$O_3$ Mixing Ratio Profile ($ppb_v$)", fontsize=18)

#     ax.set_ylabel("Altitude (km AGL)", fontsize=18)
#     ax.set_xlabel(xlabel, fontsize=18)

#     if "xlims" in kwargs.keys():
#         lim = kwargs["xlims"]
#         lims = [np.datetime64(lim[0]), np.datetime64(lim[-1])]
#         ax.set_xlim(lims)

#     if "ylims" in kwargs.keys():
#         ax.set_ylim(kwargs["ylims"][0:2])
#         ax.set_yticks(np.arange(kwargs["ylims"][0], kwargs["ylims"][1], step=kwargs["ylims"][2]))

#     if "surface" in kwargs.keys():
#         df = kwargs["surface"][0]
#         dummy = np.ones(len(df))*kwargs["surface"][1]
#         ax.scatter(df.index, dummy, c=df, cmap=ncmap, norm=nnorm)

#     converter = mdates.ConciseDateConverter()
#     munits.registry[datetime.datetime] = converter

#     ax.xaxis_date()

#     if "savefig" in kwargs.keys():
#         plt.savefig(f"{kwargs['savefig']}", dpi=600)

#     plt.show()
#     plt.rcParams.update(plt.rcParamsDefault)

#     return