# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 10:33:25 2022

@author: MRoots
"""

from pathlib import Path

import numpy as np
import pandas as pd

from datetime import datetime as dt
from datetime import timedelta as td

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib as mpl

import copy

class RWP:
    '''A Class for working with Radar Wind Profiler datasets'''
    
    def __init__(self):
        return
    
    def read_RWP(self, path, LT=-4, troubleshoot=True):
        
        """
            Designed to efficiently open, read, and import RWP datasets     
            
            Usage: Call RWP.read_RWP(your_fielpath, LT=Hours_from_UTC)
            
            Returns: Dict of files imported, keys as YYYYmmdd of datset
            
        """

        if not hasattr(path, '__iter__') or type(path) is str:
            path = [path]
            # print(path)
        
        data = {}
        
        for p in path:
            print(p)
            p = Path(p)
        
            if not p.exists():
                print(p.exists())
                print(f"{str(p)}: Does not exist on this machine...")
                return
            
            if p.is_dir():
                filepaths = [x for x in p.glob("*.cns") if x.is_file()]
            if p.is_file():
                filepaths = [p]
        
            if troubleshoot is True:
                log = open("log.RWP", "w+")
            
            
            for filepath in filepaths:
                
                if troubleshoot is True: 
                    log.write(f"{filepath}: Importing \n")
                    
                try:
                    
                    with open(filepath, "r") as f:
                        content = f.readlines()
                    
                    if troubleshoot is True: 
                        log.write(f"{filepath}: Grabbed Headers \n")
                        
                    self.LT = LT
                    self.cols = ["HT","SPD","DIR","RAD1","RAD2","RAD3","RAD4","RAD5","CNT1","CNT2","CNT3","CNT4","CNT5","SNR1","SNR2","SNR3","SNR4","SNR5"]
        
                    df = pd.read_csv(filepath, delim_whitespace=True, names=self.cols)
        
                    headers = df[df.isna().any(axis=1)]
        
                    spacing = df[df.HT == "HT"].index[1] - (df[df.HT == "HT"].index[0]+1) * 2
                    
                    if spacing == 0: 
                        log.write(f"{filepath}: Spacing == Zero")
                        continue
        
                    time = [dt.strptime(''.join(headers.loc[int(i)].to_string(header=False, index=False).replace(' ', '').split("\n")[0:7]),
                                        '%y%m%d%H%M%S0') + td(hours=self.LT) for i in np.arange(3, len(df),(spacing+11))]
        
                    df = df.dropna()
        
                    df = df[df.HT != "HT"]
                    
                    dataset = time[0].strftime("%Y%m%d")
                    data[dataset] = {"site": content[0].replace("\n", ""), "Time":time, "header": content[0:9], "loc": [np.float64(x) for x in content[2].split()]} 
        
                    for col in self.cols:
                        final_array = np.asarray(np.array([df[col]]).T, dtype=np.float64, order="C")
                        final_array[final_array>=999.0] = np.nan
                        final_array[final_array<=-999.0] = np.nan
                        data[dataset][col] = final_array.reshape((int(len(final_array)/spacing), spacing)).T
                        
                    if troubleshoot is True: 
                        log.write(f"{filepath}: Successful Import \n --- \n")
                        
                except Exception as e:
                    log.write(f"{p}: {e} \n")
                    log.write("\n --- \n")
               
        return data
    
    def LLJ_mask(self, data, DEG = [180, 270], SPD=10, **kwargs):
        
        LLJ = copy.deepcopy(data); 
        shape = (len(LLJ["HT"]), len(LLJ["Time"]))
        time = np.array([int(x.hour) for x in LLJ["Time"]])
        Time = np.full(shape, time) 
        
        LLJ["SPD"][(self.data["SPD"] < SPD)] = np.nan
        LLJ["SPD"][(self.data["DIR"] < DEG[0]) | (self.data["DIR"] > DEG[1])] = np.nan
        LLJ["SPD"][(Time > 8) & (Time < 20)] = np.nan
            
        keys = list(LLJ.keys()); keys.remove("HT"); keys.remove("Time")
        
        for i in keys:
            LLJ[i][np.isnan(LLJ["SPD"])] = np.nan
        
        return LLJ
    
    
    def plot(self, data, **kwargs):

        """
        data: dictionary object with needed vars
            - Time, HT, SPD, DIR

        **kwargs
            plot =
            
        """
        
        if "variables" in kwargs.keys():
            X, Y, SPD, DIR, VER = kwargs["variables"]
        else: 
            X, Y, SPD, DIR, VER, = (data["Time"],
                      data["HT"][:, 0],
                      data["SPD"],
                      data["DIR"],
                      data["RAD1"])

        if 'fontsize' in kwargs.keys():
            fontsize = kwargs["fontsize"]
        else:
            fontsize = 14

        if 'plt' in kwargs.keys():
            fig, ax1, ax2, ax3 = kwargs["plt"]

        else:
            fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(15,14))

    
        if "LLJ_mask" in kwargs.keys():
            LLJ_mask = self.LLJ_mask()
            X, Y, SPD, DIR, VER = (LLJ_mask["Time"],
                      LLJ_mask["HT"][:, 0],
                      LLJ_mask["SPD"],
                      LLJ_mask["DIR"],
                      LLJ_mask["RAD1"])
        
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

        if "xticks" in kwargs.keys():
            if kwargs["xticks"] == "HR":
                ax1.xaxis.set_major_locator(mdates.HourLocator(interval = 4))
                ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
                ax1.xaxis.set_minor_locator(mdates.HourLocator(interval = 1))

                ax2.xaxis.set_major_locator(mdates.HourLocator(interval = 4))
                ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
                ax2.xaxis.set_minor_locator(mdates.HourLocator(interval = 1))

                ax3.xaxis.set_major_locator(mdates.HourLocator(interval = 4))
                ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
                ax3.xaxis.set_minor_locator(mdates.HourLocator(interval = 1))
        else:
            ax1.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax1.xaxis.get_major_locator()))
            ax2.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax2.xaxis.get_major_locator()))
            ax3.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax2.xaxis.get_major_locator()))

        return ax1, ax2, ax3

##############################################################################
#
#   General Utility Functions (DATAS objs)
#
##############################################################################


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
    dir_path = Path(r"./data/beltsville")
    figpath = dir_path / "figures"; figpath.mkdir(parents=True, exist_ok=True)
    rwp = RWP()
    data = rwp.read_RWP(dir_path, LT=-4, troubleshoot=True)
    
    for key in data.keys():
        rwp.plot(data[key])
        plt.savefig(f"{figpath / key}.png", dpi=300)
        plt.close()