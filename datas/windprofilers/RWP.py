# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 10:33:25 2022

@author: Magnolia
"""

import pandas as pd
import numpy as np

from datetime import datetime as dt
from datetime import timedelta as td

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import matplotlib as mpl


class RWP:
    '''class for working with Radar Wind Profiler datasets'''

    def __init__(self, filepaths):

        self.cols = ["HT","SPD","DIR","RAD1","RAD2","RAD3","RAD4","RAD5","CNT1","CNT2",
        "CNT3","CNT4","CNT5","SNR1","SNR2","SNR3","SNR4","SNR5"]
        self.data = self.read_RWP(filepaths)

    def read_RWP(self, filepaths, LT=-4):

        self.LT = LT

        df = pd.concat([pd.read_csv(filepath, delim_whitespace=True, names=self.cols) for filepath in filepaths], ignore_index=True)

        headers = df[df.isna().any(axis=1)]

        spacing = df[df.HT == "HT"].index[1] - (df[df.HT == "HT"].index[0]+1) * 2

        time = [dt.strptime(''.join(headers.loc[int(i)].to_string(header=False, index=False).replace(' ', '').split("\n")[0:7]),
                            '%y%m%d%H%M%S0') + td(hours=self.LT) for i in np.arange(3, len(df),(spacing+11))]

        df = df.dropna()

        df = df[df.HT != "HT"]

        data = {"Time": time}

        for col in self.cols:
            final_array = np.asarray(np.array([df[col]]).T, dtype=np.float64, order="C")
            final_array[final_array>=999.0] = np.nan
            final_array[final_array<=-999.0] = np.nan
            data[col] = final_array.reshape((int(len(final_array)/spacing), spacing)).T

        return data

    def plot(self, **kwargs):

        """
        data: dictionary object with needed vars
            - Time, HT, SPD, DIR

        **kwargs
            plot =
        """

        if 'fontsize' in kwargs.keys():
            fontsize = kwargs["fontsize"]
        else:
            fontsize = 14

        if 'plt' in kwargs.keys():
            fig, ax1, ax2, ax3 = kwargs["plt"]

        else:
            fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(15,14))



        X, Y, SPD, DIR, VER = (self.data["Time"],
                              self.data["HT"][:, 0],
                              self.data["SPD"],
                              self.data["DIR"],
                              self.data["RAD1"])

        ncmap, nnorm = utils.colormaps("cardinal8")

        # SPEED
        im1 = ax1.contourf(X, Y, SPD, levels=np.arange(0, 21, 2.5), cmap="turbo")
        cbar1 = fig.colorbar(im1,ax=ax1, ticks=np.arange(0, 50, 5), pad=0.01,
                      spacing='proportional', label="Horizontal Velocity (m/s)")
        cbar1.set_label(label="Horizontal Velocity (m/s)", size=fontsize)

        ax1.set_ylabel("Altitude (km AGL)", fontsize=fontsize)

        # DIRECTION
        ax2.contourf(X, Y, DIR, cmap=ncmap)

        ax2.set_ylabel("Altitude (km AGL)", fontsize=fontsize)
        # ax2.set_xlabel(f"Local Time (UTC {self.LT})", fontsize=fontsize)




        # VERTICAL
        if "ver_levels" in kwargs.keys():
            ver_levels = kwargs["ver_levels"]
        else: 
            ver_levels = np.arange(-5, 5.1, 1)
            
        im2 = ax3.contourf(X, Y, VER, levels=ver_levels, cmap="turbo")
        cbar3 = fig.colorbar(im2, ax=ax3, ticks=np.arange(-5, 5.1, 1), pad=0.01,
                      spacing='proportional', label="Vertical Velocity (m/s)")
        cbar3.set_label(label="Vertical Velocity (m/s)", size=fontsize)

        ax3.set_ylabel("Altitude (km AGL)", fontsize=fontsize)



        if "contour" in kwargs.keys():
            if kwargs["contour"][0] == True:
                cp1 = ax1.contour(X, Y, SPD, levels=np.arange(0, 21, 2.5), colors='black', linestyles='solid')
                ax1.clabel(cp1, inline=True, fmt='%.0f', fontsize=10, rightside_up=True)
            if kwargs["contour"][1] == True:
                ax2.contour(X, Y, DIR, colors='black', linestyles='solid')

        if "degrees" in kwargs.keys() and kwargs["degrees"]:
            cbar2 = fig.colorbar(mpl.cm.ScalarMappable(cmap=ncmap, norm=nnorm), ax=ax2, ticks=[0, 90, 180, 270, 360], pad=0.01, spacing='proportional')
            cbar2.set_label(label="Wind Direction (degrees)", size=fontsize)

        else:
            cbar2 = fig.colorbar(mpl.cm.ScalarMappable(cmap=ncmap, norm=nnorm), ax=ax2, ticks=[22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5], pad=0.01,
                      spacing='proportional')
            cbar2.set_label(label="Wind Direction", size=fontsize)
            cbar2.ax.set_yticklabels(['NNE', 'ENE', 'ESE', 'SSE', 'SSW', 'WSW', 'WNW', 'NNW'])

        if "title" in kwargs.keys():
                ax1.set_title(kwargs["title"], fontsize=20)
        else: ax1.set_title(r"Radio Wind Profiler (RWP)", fontsize=20)

        if "xlims" in kwargs.keys():
            lim = kwargs["xlims"]
            lims = [np.datetime64(lim[0]), np.datetime64(lim[-1])]
            ax1.set_xlim(lims)
            ax2.set_xlim(lims)
            ax3.set_xlim(lims)

        if "ylims" in kwargs.keys():
            ax1.set_ylim(kwargs["ylims"])
            ax2.set_ylim(kwargs["ylims"])
            ax3.set_ylim(kwargs["ylims"])

        else:
            ax1.set_ylim([0, 3.01]); ax2.set_ylim([0, 3.01]), ax3.set_ylim([0, 3.01])

        if "yticks" in kwargs.keys():
            ax1.set_yticks(kwargs["yticks"])
            ax2.set_yticks(kwargs["yticks"])
            ax3.set_yticks(kwargs["yticks"])

        plt.setp(ax1.get_yticklabels(), fontsize=fontsize)
        plt.setp(ax1.get_xticklabels(), fontsize=fontsize)

        plt.setp(ax2.get_yticklabels(), fontsize=fontsize)
        plt.setp(ax2.get_xticklabels(), fontsize=fontsize)

        plt.setp(ax3.get_yticklabels(), fontsize=fontsize)
        plt.setp(ax3.get_xticklabels(), fontsize=fontsize)

        cbar1.ax.tick_params(labelsize=fontsize)
        cbar2.ax.tick_params(labelsize=fontsize)
        cbar3.ax.tick_params(labelsize=fontsize)

        ax1.grid(True, axis=('both'))
        ax2.grid(True, axis=('both'))
        ax3.grid(True, axis=('both'))

        ax1.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax1.xaxis.get_major_locator()))
        ax2.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax2.xaxis.get_major_locator()))
        ax3.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax2.xaxis.get_major_locator()))
        
        return ax1, ax2


class utils:
    '''class for utility functions'''
    def __init__(self):
        return

    def colormaps(cmap_name):
        custom_maps = {"cardinal8":{"cmap":(mpl.colors.ListedColormap(['red', 'darkorange', "gold", "forestgreen", "turquoise", "mediumblue", "darkviolet", "deeppink"])), "bounds": [0, 45, 90, 135, 180, 225, 270, 315, 360]}}

        ncmap = custom_maps[cmap_name]["cmap"]
        bounds = custom_maps[cmap_name]["bounds"]
        ncmap.set_under([1,1,1])
        ncmap.set_over([0,0,0])

        nnorm = mpl.colors.BoundaryNorm(bounds, ncmap.N)
        return ncmap, nnorm


if __name__ == "__main__":

    rwp_filepaths = [x for x in Path(r"..\data\RWP").glob("Beltsville\*.cns") if x.is_file()]

    rwp = RWP(rwp_filepaths)

#%%
    X, Y, Z = (rwp.data["Time"], rwp.data['HT'], rwp.data["RAD1"])
    plt.figure()
    plt.pcolormesh(X, Y, Z)

    """
    data: dictionary object with needed vars
        - Time, HT, SPD, DIR

    **kwargs
        plot =
    """

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(15,14))

    X, Y, SPD, DIR, VIR = (self.data["Time"],
                          self.data["HT"][:, 0],
                          self.data["SPD"],
                          self.data["DIR"],
                          self.data["RAD1"])

    ncmap, nnorm = utils.colormaps("cardinal8")

    im1 = ax1.contourf(X, Y, SPD, levels=np.arange(0, 21, 2.5), cmap="turbo")
    cbar1 = fig.colorbar(im1,ax=ax1, ticks=np.arange(0, 50, 5), pad=0.01,
                  spacing='proportional', label="Wind Speed (m/s)")

    ax2.contourf(X, Y, DIR, cmap=ncmap)
    ax2.contourf(X, Y, DIR, cmap=ncmap)

    cbar1.set_label(label="WindSpeed (m/s)", size=fontsize)

    if "contour" in kwargs.keys():
        if kwargs["contour"][0] == True:
            cp1 = ax1.contour(X, Y, SPD, levels=np.arange(0, 21, 2.5), colors='black', linestyles='solid')
            ax1.clabel(cp1, inline=True, fmt='%.0f', fontsize=10, rightside_up=True)
        if kwargs["contour"][1] == True:
            ax2.contour(X, Y, DIR, colors='black', linestyles='solid')

    if "degrees" in kwargs.keys() and kwargs["degrees"]:
        cbar2 = fig.colorbar(mpl.cm.ScalarMappable(cmap=ncmap, norm=nnorm), ax=ax2, ticks=[0, 90, 180, 270, 360], pad=0.01, spacing='proportional')
        cbar2.set_label(label="Wind Direction (degrees)", size=fontsize)

    else:
        cbar2 = fig.colorbar(mpl.cm.ScalarMappable(cmap=ncmap, norm=nnorm), ax=ax2, ticks=[22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5], pad=0.01,
                  spacing='proportional')
        cbar2.set_label(label="Wind Direction", size=fontsize)
        cbar2.ax.set_yticklabels(['NNE', 'ENE', 'ESE', 'SSE', 'SSW', 'WSW', 'WNW', 'NNW'])

    if "title" in kwargs.keys():
            ax1.set_title(kwargs["title"], fontsize=20)
    else: ax1.set_title(r"Radio Wind Profiler (RWP)", fontsize=20)

    ax1.set_ylabel("Altitude (km AGL)", fontsize=fontsize)
    ax2.set_ylabel("Altitude (km AGL)", fontsize=fontsize)
    ax2.set_xlabel(f"Local Time (UTC {self.LT})", fontsize=fontsize)

    if "xlims" in kwargs.keys():
        lim = kwargs["xlims"]
        lims = [np.datetime64(lim[0]), np.datetime64(lim[-1])]
        ax1.set_xlim(lims)
        ax2.set_xlim(lims)

    if "ylims" in kwargs.keys():
        ax1.set_ylim(kwargs["ylims"])
        ax2.set_ylim(kwargs["ylims"])
    else:
        ax1.set_ylim([0, 3.01]); ax2.set_ylim([0, 3.01])

    if "yticks" in kwargs.keys():
        ax1.set_yticks(kwargs["yticks"])
        ax2.set_yticks(kwargs["yticks"])

    plt.setp(ax1.get_yticklabels(), fontsize=fontsize)
    plt.setp(ax1.get_xticklabels(), fontsize=fontsize)
    plt.setp(ax2.get_yticklabels(), fontsize=fontsize)
    plt.setp(ax2.get_xticklabels(), fontsize=fontsize)
    cbar1.ax.tick_params(labelsize=fontsize)
    cbar2.ax.tick_params(labelsize=fontsize)

    ax1.grid(True, axis=('both'))
    ax2.grid(True, axis=('both'))

    ax1.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax1.xaxis.get_major_locator()))
    ax2.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax2.xaxis.get_major_locator()))

    # return ax1, ax2