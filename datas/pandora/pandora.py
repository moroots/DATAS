 # -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 16:40:03 2020

@author: Magnolia

Pandora
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import matplotlib.dates as mdates

class pandonia:
    def clean(df, value=12):
            df.loc[df[11] >= value, [7, 8, 9, 10]] = np.nan
            # df.loc[df[22] >= 1, [7:10]] = np.nan
            df.loc[df[18] >= value, [14, 15, 16, 17]] = np.nan
            return df

    def flt_by_date(df, date1, date2):
        if df.empty: print('Input dataframe is empty'); return
        mask = (df[0] >= date1) & (df[0] <= date2)
        uL = df.loc[mask]
        if uL.empty: print('Specified dates are not found \n',
                                    f'Dataframe only provides {df[0].iloc[0]} to {df[0].iloc[-1]}'); return
        return uL

    def flt(df, column, val_1, val_2):
        if df.empty: print('Input dataframe is empty'); return
        mask = (df[column] >= val_1) & (df[column] <= val_2)
        uL = df.loc[mask]
        if uL.empty: print('Specified values are not found \n'); return
        return uL

    def plot(dataframe, mol=None, flt='on', title=None, savpath=None):
        '''### Description ###
        # dataframe -> enter pandas data frame
        # start     -> enter starting date (or datetime) as string
        # stop      -> ente stoping date (or datetime) as string
        # fig'''

        if mol and type(mol) is not str: print('Molecule type must be a string value')
        if title and type(title) is not str: print('Title type must be a string value')
        if savpath and type(savpath) is not str: print('Save-Path must be a string value')

        ### Output => dateframe with
        uL = dataframe
        uL[7].replace(-999, np.nan)

        if flt == 'on':
            uL.loc[uL[11] >= 12, [7, 8, 3, 4, 14, 15]] = np.nan
            uL.loc[uL[22] >= 1, [7, 8, 3, 4, 14, 15]] = np.nan

        uL1 = uL[0].iloc[0].strftime('%Y%m%dT%H%M')
        uL2 = uL[0].iloc[-1].strftime('%Y%m%dT%H%M')
        savnam=f'{uL1}_{uL2}'

        if not title:
            title = savnam

        fig, (ax1, ax3) = plt.subplots(2, 1)
        color = 'blue'
        ax1.plot(uL[0], uL[8], '.', markersize=1, color='b')
        ax1.set_title(f"{title}")
        ax1.set_ylabel('Uncertainty (DU)', color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True)

        ax2 = ax1.twinx()

        color = 'red'
        ax2.plot(uL[0], uL[7], '.', markersize=1, color=color, label='Data')
        ax2.set_ylabel('Total Column (DU)', color=color)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("(%Hh)"))
        ax2.tick_params(axis='y', labelcolor=color)
        if type(mol) is str: ax2.legend([mol])


        fig.tight_layout()  # otherwise the right y-label is slightly clipped

        color = 'tab:blue'
        ax3.plot(uL[0], uL[3],'.', markersize=1, color=color)
        ax3.set_xlabel('Datetime UTC')
        ax3.set_ylabel('SZA (degrees)', color=color)
        ax3.tick_params(axis='y', labelcolor=color)

        ax4 = ax3.twinx()

        color = 'green'
        ax4.plot(uL[0], uL[4],'.', markersize=1, color=color)
        ax4.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        ax4.set_ylabel('SAA (degrees)', color=color)
        ax4.tick_params(axis='y', labelcolor=color)
        ax3.grid(True)
        if savpath: fig.savefig(f"{savpath}\QuickPlot_{savnam}.png", dpi=600)
        if not savpath: fig.savefig(f"QuickPlot_{savnam}.png", dpi=600)
        plt.show()
        return uL

    def plot_by_date(dataframe, start, stop, mol=None, flt='off', title=None, savpath=None):
        '''### Description ###
        # dataframe -> enter pandas data frame
        # start     -> enter starting date (or datetime) as string
        # stop      -> ente stoping date (or datetime) as string
        # fig

        ### Output => dateframe with'''

        uL = pandonia.flt_by_date(dataframe, start, stop); uL = uL.reset_index()

        uL[7].replace(-999, np.nan)

        if flt == 'on':
            uL.loc[uL[11] >= 12, [7, 8, 3, 4]] = np.nan
            uL.loc[uL[22] >= 1, [7, 8, 3, 4]] = np.nan
        title1 = title
        savpath1 = savpath
        mol1 = mol
        pandonia.plot(uL, mol=mol1, title=title1, savpath=savpath1)
        return uL

    def importing(Path, par=None, date_start=None, date_stop=None):
        uL = {}
        uL_head = {}
        print(f'Importing Panodnia Data from {Path} \n')
        for filename in glob.glob(os.path.join(Path, '*.txt')):
            """Search for the given string in file and return lines containing that string,
            along with line numbers"""
            nlines = 0
            cells = []
            # Open the file in read only mode
            with open(filename, 'r') as f:
                for line in f:   # Read all lines in the file one by one
                    nlines += 1
                    if '---' in line:
                        cells.append(nlines-1)    # If yes, then add the line number
                f.close()
            with open(os.path.join(Path, filename), 'r') as f: # open in readonly mode
                uL_head[filename.split('\\')[-1]] = pd.read_csv(f, sep="\n",
                                                                              header=None, nrows=cells[1], low_memory=False)
                f.close()
            with open(os.path.join(Path, filename), 'r') as f:
                uL[filename.split('\\')[-1]] = pd.read_csv(f, sep=" ", parse_dates=[0], header=None, skiprows=cells[1]+1, low_memory=False)
                print(filename.split('\\')[-1], '\t -> Loaded')
                f.close()

        name = list(uL.keys())

        if par != None:
            if 'filter by date' in par and date_start !=None:
                if date_start & date_stop:
                    for i in name:
                        uL[i] = pandonia.flt_by_date(uL[i], date_start, date_stop)
            if 'plot' in par:
                if 'filter on' in par: flt1='on'
                for i in name:
                    uL[i] = pandonia.plot(uL[i], flt=flt1)

        return uL, uL_head, name

if __name__ == '__main__':

    # from bs4 import BeautifulSoup
    # import requests
    # from pathlib import Path
    # import os

    # # def data_pandonia_database():


    # def data_pandonia_download(location, instrument, level, file, destination_path):
    #     if location is not type([]): location = [location]
    #     if instrument is not type([]): instrument = [instrument]
    #     if level is not type([]): level = [level]

    #     for loc in location:
    #         for instr in instrument:
    #             for ft in filetype:
    #                 if file is not type([]):
    #                     file = [file]
    #                 elif file == 'all':
    #                     path = "http://data.pandonia-global-network.org/{loc}/{instr}/{ft}/{fil}"
    #                     page = requests.get(file_url)

    #                 for fil in file:
    #                     path = "http://data.pandonia-global-network.org/{loc}/{instr}/{ft}/{fil}"
    #                     file_url = parent + path
    #                     page = requests.get(file_url)

    #                     savnam = destination_path / Path(path)
    #                     os.makedirs(os.path.dirname(savnam), exist_ok=True)
    #                     with open(savnam, 'wb') as f:
    #                         f.write(page.content)
    #                         f.close()
    #                     print(f"Downloaded -> {fil}")

    #     # if 'all' in file

    #     return

    # destination_path = r"C:\Users\Magnolia\OneDrive - UMBC\Research\Data\Smoke Case 20200308\Pandora"
    # location = r"GreenbeltMD"
    # instrument = r"Pandora32s1"
    # filetype = r"L1"
    # file = r"Pandora32s1_GreenbeltMD_20171222_L1_smca1c1p1-7.txt.bz2"
    # data_pandonia_download(location, instrument, filetype, file, destination_path)



#%%
# import bz2

# savnam = f"{destination_path}/{location}/{instrument}/{filetype}/{file}"
# with bz2.open(savnam, "rb") as f:
#     # Decompress data from file
#     content = f.read()
# with open(savnam.strip('.bz2'), 'wb') as fil:
#     fil.write(content)

#%%
