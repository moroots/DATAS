# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 15:39:00 2020

@author: meroo

Sondes

"""

import os
import numpy as np
import glob as glob
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

class CFH:
    def clean(df, column):
        df.loc[df['O3 Mr[ppmv]'] > 99^9] = np.nan
        return df

    def importing(data_path, clean=False, plot=False, log_scale=False, save_name='CFH_Ozone_Sonde'):
        current = os.getcwd()
        uL = {}
        uL_head = {}
        print(f'Importing CFH Sonde data from {data_path} \n')
        for filename in glob.glob(os.path.join(data_path, '*.dat')):
            """Search for the given string in file and return lines containing that string,
            along with line numbers"""
            with open(filename, 'r') as f:
                lines_to_read = [1]
                for position, line in enumerate(f):
                    if position in lines_to_read:
                        start = int(line.split('= ')[-1])

            with open(os.path.join(data_path, filename), 'r') as f:
                uL_head[filename.split('\\')[-1]] = pd.read_csv(f, sep="\n",
                    header=None, nrows=start-3, low_memory=False)
                f.close()

            with open(os.path.join(data_path, filename), 'r') as f:
                uL_columns = pd.read_csv(f, sep=",", header=None, skiprows=start-2, nrows=2, skipinitialspace=True, low_memory=False)
                uL_c = list(uL_columns.loc[0] + uL_columns.loc[1])
                f.close()

            with open(os.path.join(data_path, filename), 'r') as f:
                uL[filename.split('\\')[-1]] = pd.read_csv(f, sep=",", names=uL_c, skiprows=start, low_memory=False)
                if clean == True: uL[filename.split('\\')[-1]] = CFH.clean(uL[filename.split('\\')[-1]], 'O3 Mr[ppmv]')
                print(filename.split('\\')[-1], '\t -> Loaded')
                f.close()

        names = list(uL.keys())


        def plot_Ozone():
            plt.style.use('default')
            plt.figure
            fig, ax = plt.subplots()
            for name in names:
                col = list(uL_head[name].columns)
                p = str(uL_head[name][uL_head[name].str.contains('Total ozone column (Dobson)', regex=False)])
                DU = p.split('= ')[-1]; DU = DU.split('(')[0]; DU = DU.replace(" ", "")
                lgnd = f'{uL_head[name].iloc[4]}'; lgnd = lgnd.split('= ')[-1]; lgnd = lgnd.split('\n')[0]
                plt.plot(uL[name]['O3 Mr[ppmv]']*1000, uL[name]['Press[hPa]'], label=f'{lgnd}: $TCO_3$ ({DU} DU)')
            plt.ylabel('Pressure (hPa)')
            plt.xlabel('Ozone (ppbv)')
            if log_scale is True: plt.yscale('log')
            plt.gca().invert_yaxis()
            plt.grid(True, which='major')
            plt.grid(True, which='minor')
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            plt.legend(fontsize=8)
            plt.title('CFH Ozone Sondes')
            if type(save_name) is str: plt.savefig(f'{save_name.replace(" ", "_")}.png', dpi=600)
            plt.show()
            return

        if plot is True: plot_Ozone()
        os.chdir(current)
        return uL, uL_head, names, uL_c
