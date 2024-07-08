# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 14:13:06 2024

@author: Maurice Roots

Classes for Working with Dataset:

"""

import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib as mpl

from datetime import datetime as dt
from datetime import timedelta as td

class RWP:
    '''A Class for working with Radar Wind Profiler datasets'''

    def __init__(self):

        return
    
    @staticmethod
    def read(path, LT=0, troubleshoot=True):

        """
            Designed to efficiently open, read, and import RWP datasets

            Usage: Call RWP.read_RWP(your_fielpath, LT=Hours_from_UTC)

            Returns: Dict of files imported, keys as YYYYmmdd of datset

        """
        cols = ["HT","SPD","DIR","RAD1","RAD2","RAD3","RAD4","RAD5","CNT1","CNT2",
        "CNT3","CNT4","CNT5","SNR1","SNR2","SNR3","SNR4","SNR5"]
        
        if not hasattr(path, '__iter__') or type(path) is str:
            path = [path]
            # print(path)

        data = {}

        for p in path:
            # print(p)
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
                log = open("log_import.RWP", "w+")


            for filepath in filepaths:

                if troubleshoot is True:
                    log.write(f"{filepath}: Importing \n")

                try:

                    with open(filepath, "r") as f:
                        content = f.readlines()

                    if troubleshoot is True:
                        log.write(f"{filepath}: Grabbed Headers \n")

                    df = pd.read_csv(filepath, sep='\s+', names=cols)

                    headers = df[df.isna().any(axis=1)]

                    spacing = df[df.HT == "HT"].index[1] - (df[df.HT == "HT"].index[0]+1) * 2

                    if (spacing == 0) and (troubleshoot is True):
                        log.write(f"{filepath}: Spacing == Zero")
                        continue

                    time = [dt.strptime(''.join(headers.loc[int(i)].to_string(header=False, index=False).replace(' ', '').split("\n")[0:7]),
                                        '%y%m%d%H%M%S0') + td(hours=LT) for i in np.arange(3, len(df),(spacing+11))]

                    df = df.dropna()

                    df = df[df.HT != "HT"]

                    dataset = time[0].strftime("%Y%m%d")
                    data[dataset] = {"site": content[0].replace("\n", ""), "Time":time, "header": content[0:9], "loc": [np.float64(x) for x in content[2].split()]}

                    for col in cols:
                        final_array = np.asarray(np.array([df[col]]).T, dtype=np.float64, order="C")
                        final_array[final_array>=999.0] = np.nan
                        final_array[final_array<=-999.0] = np.nan
                        data[dataset][col] = final_array.reshape((int(len(final_array)/spacing), spacing)).T

                    if troubleshoot is True:
                        log.write(f"{filepath}: Successful Import \n --- \n")

                except Exception as e:
                    log.write(f"{filepath}: Error {e}\n")
                    print(f"Errors occurred {filepath.name}. Check 'log.RWP' for details.")
                    
        return data
    
    @staticmethod
    def colormaps(cmap_name):
        c82_listed_cmap = mpl.colors.ListedColormap(['red', 
                                   'darkorange', 
                                   "gold", 
                                   "forestgreen", 
                                   "turquoise", 
                                   "mediumblue", 
                                   "darkviolet", 
                                   "deeppink"])
        
        c82_bounds = [0, 45, 90, 135, 180, 225, 270, 315, 360]
        
        c81_listed_cmap = mpl.colors.ListedColormap(['darkred',
                                                     'deeppink',
                                                     'darkviolet',
                                                     'mediumblue',
                                                     'forestgreen',
                                                     'turquoise',
                                                     'gold',
                                                     'darkorange',
                                                     'darkred'])
        
        c81_bounds = [0, 22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5, 361]
        
        
        custom_maps = {"cardinal8_02":{"cmap":(c82_listed_cmap), "bounds": c82_bounds},
                       "cardinal8_01":{"cmap":(c81_listed_cmap), "bounds": c81_bounds}}

        ncmap = custom_maps[cmap_name]["cmap"]
        bounds = custom_maps[cmap_name]["bounds"]
        ncmap.set_under([1,1,1])
        ncmap.set_over([0,0,0])

        nnorm = mpl.colors.BoundaryNorm(bounds, ncmap.N)
        return ncmap, nnorm
    


if __name__ == "__main__":
    
    # For only several files
    filepaths = [x for x in Path("../data").glob("*.cns")]
    Few_Files = RWP.read(filepaths[:10])
    
    # For LOTS of files: its best to do seperate calls
    # Depends on the memory capacity of your machine, and what you plan to do next
    # Its also easier to multi-process this way: BUT use troubleshoot=False
    filepaths = [x for x in Path("../data").glob("*.cns")]
    Many_Files = {}
    for filepath in filepaths: 
        temp = RWP.read(filepaths, troubleshoot=False); key = list(temp.keys())[0]
        Many_Files[key] = temp[key]