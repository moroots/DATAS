# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 18:30:32 2022

@author: meroo
"""

# %% Import Packages

import matplotlib.pyplot as plt
import numpy as np
import time
import matplotlib.dates as mdates
import xarray as xr
import os
from glob import glob
from matplotlib.colors import LogNorm
import pandas as pd
from scipy.signal import savgol_filter
from scipy.signal import convolve

from scipy.signal import find_peaks
from pathlib import Path

from numba import jit

from metpy.units import units
#%% Function Space

def importing_ceilometer(FilePaths, **kwargs):
    data = {} # all data will be stored into a nested dictionary
    files = {}
    FilePaths = [Path(filePath) for filePath in FilePaths] # converting to Pathlib objects

    for filePath in FilePaths:
        if filePath.is_file() is False:
            print('PATH IS NOT FOUND ON MACHINE')
            return
        
        fileName = filePath.name
        data[fileName] = {} # Nested Dictionary
        with xr.open_dataset(filePath) as file: # importing data as a pyhdf obj
            data[fileName]["time"] = file.time.values
            data[fileName]["time"] = file.time.values
            data[fileName]["dateNum"] = [mdates.date2num(t) for t in data[fileName]["time"]]
            data[fileName]["range"] = file.range.values
            data[fileName]["beta_raw"] = file.beta_raw.values
            data[fileName]["beta_raw"][data[fileName]["beta_raw"] < 0] = 0
            data[fileName]["beta_raw"] = data[fileName]["beta_raw"].T
            data[fileName]["instrument_pbl"] = file.pbl.values
            data[fileName]["lat_lon_alt"] = [file.longitude.values, file.latitude.values, file.altitude.values]
    
            if "vars" in kwargs.keys():
                for var in kwargs["vars"]:
                    data[fileName][var] = file[var].values
            
            data[fileName]["datasets"] = list(file.keys())
            files[fileName] = file
            
    return data, files

# Calculate Haar Wavelet at only one altitude 'b'
@jit(nopython=True)
def haar(profile,alts,a,b):

    # Zero Array to fill in Wavelet
    wavelet = np.zeros(alts.shape)

    # Haar Wavelet (see eqn 1 screenshot)
    wavelet[np.int(b - a):np.int(b)] = -1
    wavelet[np.int(b):np.int(b + a)] = 1
    wavelet[b] = 0

    # Calculate integral (sum f(z)*h(z-b/a) over all z)
    w_ab = np.nansum(profile * wavelet)

    return w_ab

# Function to loop altitudes
@jit(nopython=True)
def covariance_profile(profile,a,alts):

    # Initialize covariance array
    w = np.zeros(alts.shape[0])

    # Height boundaries for wavelet
    z_top = np.int(a)+1
    z_bot = profile.shape[0]-np.int(a)-1

    # Calculate w(a,b) for each altitude
    for z in list(range(z_top, z_bot)):

        # Calculate covariance coefficients and fill covariance array
        w[z] = haar(profile, alts, a, z) / a

    return w

# Applying HCWT on backscatter
@jit(nopython=True)
def pblh_timeseries(backscatter, alts, a):
    start_time = time.time()
    pbl_top = []
    pbl_bottom = []
    hcwt = {}
    for t in range(0, len(backscatter[0,:])):
        profile = backscatter[:, t]
        w = covariance_profile(profile,a,alts);
        uL_top = list(np.where(w == max(w))[0]);
        top = int(uL_top[0])
        bottom =  int(list(np.where(w == min(w[top:-1]))[0])[0])
        pbl_top += [alts[top]]
        pbl_bottom += [alts[bottom]]
        hcwt[t] = w
    end_time = time.time()
    print(f"Execuded in {round(end_time - start_time, 3)} seconds")
    return {"top": pbl_top, "bottom": pbl_bottom, "entrainment_depth": abs(np.mean(pbl_top) - np.mean(pbl_bottom)), "hcwt":hcwt}

# Savistsky-Golay Smoothing
def savgol(backscatter):
    smoothed = np.zeros(backscatter.shape)
    # print(smoothed.shape)
    for t in range(0, len(smoothed[0,:])):
        profile_interp = pd.DataFrame({"flt_backscatter":backscatter[:,t]})
        profile_interp["intp_backscatter"] = profile_interp["flt_backscatter"].interpolate(method="linear")
        smoothed[:, t] = savgol_filter(profile_interp["intp_backscatter"], window_length=11, polyorder=3)
    return smoothed

def moving_average(data, window):
    df = pd.DataFrame(data["backscatter"].T, index=data["time"], columns=data["alt"])
    df_avg = df.rolling(window).mean()
    df_avg=df_avg.to_numpy()
    return df_avg.T

#%% Importing the data

FilePaths = [r"C:/Users/meroo/OneDrive - UMBC/Research/Data/Celiometer/test_data/20210518_Catonsville-MD_CHM160112_000.nc"]
data, files = importing_ceilometer(FilePaths)
keys = list(data.keys())

#%% Testbed
start_time = time.time()

a = 5
alt_stop = np.where(data[keys[0]]["range"]>=8000)[0][0]
alt_start = np.where(data[keys[0]]["range"]>=100)[0][0]

smoothed = savgol(data[keys[0]]["beta_raw"][alt_start:alt_stop, :])

#%%

a = [1, 10]
a_vals = np.arange(a[0], a[1])
# pblh = np.zeros((data["backscatter"][alt_start:alt_stop, :]).shape)
c = np.zeros((len(data[keys[0]]["beta_raw"][alt_start:alt_stop]), len(a_vals)))
alts = data[keys[0]]["range"][alt_start:alt_stop]
c_avg = np.zeros((data[keys[0]]["beta_raw"][alt_start:alt_stop, :]).shape)

for i in range(data[keys[0]]["beta_raw"][alt_start:alt_stop, :].shape[1]):
    for a, j in zip(a_vals, range(0, len(c)+1)):
        c[:,j]= covariance_profile(data[keys[0]]["beta_raw"][alt_start:alt_stop, i], a, alts)
        print(f"index: {i}, a = {a}")
    c_avg[:, i] = np.mean(c, axis=1)

end_time = time.time()
print(f"Execuded in {round(end_time - start_time, 3)} seconds")

#%% Finding Peaks

# test_spot = 5000
# alt = data["alt"][alt_start:alt_stop]
# x = c_avg[:, test_spot]

# peaks_top, properties = find_peaks(x, height=0, threshold=500, distance=50, prominence=(5e3, 5e6))
# peaks_bottom, properties = find_peaks(-1*x, height=0, threshold=100, distance=50, prominence=(5e3, 5e6))

# plt.figure()
# plt.plot(x, alt)
# plt.plot(x[peaks_top], alt[peaks_top], '^', color="tab:red")
# plt.plot(x[peaks_bottom], alt[peaks_bottom], 'v', color="tab:red")
# plt.xlim(-1e6, 1e7)
# plt.ylabel("Altitude")
# plt.title(f"20210517_Catonsville-MD_CHM160112_000.nc: t={test_spot}")
# plt.show()

# t_top = np.full(shape=alt[peaks_top].shape, fill_value=data["time"][test_spot])
# t_bottom = np.full(shape=alt[peaks_bottom].shape, fill_value=data["time"][test_spot])

# plt.figure(figsize=(12, 6), constrained_layout=True)
# im = plt.pcolormesh(data["time"],data["alt"],data["backscatter"],cmap='jet', norm=LogNorm())
# plt.axvline(data["time"][test_spot+30], linestyle='--', color="k")
# plt.axvline(data["time"][test_spot-30], linestyle='--', color="k")
# plt.plot(t_top, alt[peaks_top], 'v', color="pink")
# plt.plot(t_bottom, alt[peaks_bottom], '^', color="pink")
# plt.ylim([0,6000])
# cbar = plt.colorbar()
# im.set_clim(vmin=10**3.5, vmax=10**8.5)
# cbar.set_label('Aerosol Backscatter')
# plt.xlabel('Datetime (UTC)')
# plt.ylabel('Altitude (m AGL)')
# plt.title("20210517_Catonsville-MD_CHM160112_000.nc")
# plt.show()

#%% PLotting Pblh

peaks_bottom = np.zeros((len(c_avg[0,:]), 50))
for i in range(0, len(c_avg[0,:])):
    peaks, properties = find_peaks(-1*c_avg[:, i], height=0, threshold=500, distance=50, prominence=(5e3, 5e4))
    order_test = sorted(((value, index) for index, value in enumerate(properties["prominences"])), reverse=True)
    # print(peaks)
    for j in range(0, len(peaks)):
        peaks_bottom[i, j] = peaks[order_test[j][1]]
        # print(peaks[j])
    # print(i)
peaks_bottom = peaks_bottom.astype(int)

#%%

df = pd.read_csv(r"C:/Users/meroo/OneDrive - UMBC/Research/Code/Python/FromKylie/New folder/vanessa_lufft_pbl.csv", header=None)

#%%

# plt.figure(figsize=(12, 6), constrained_layout=True)
# # im = plt.pcolormesh(data["time"],data["alt"],data["backscatter"],cmap='jet', norm=LogNorm())
# # im = plt.pcolormesh(data["time"],data["alt"],df_avg.T,cmap='jet', norm=LogNorm())
# # im = plt.pcolormesh(data["time"],data["alt"][alt_start:alt_stop],smoothed,cmap='jet', norm=LogNorm())
# # im = plt.pcolormesh(data["time"],data["alt"],test,cmap='jet', norm=LogNorm())
# # plt.plot(data["time"], alt[peaks_bottom[:, 2]], '^', color="pink")
# # plt.plot(data["time"], df[1], 'x', color="k")
# plt.ylim([0,6000])
# cbar = plt.colorbar()
# im.set_clim(vmin=10**3.5, vmax=10**8.5)
# cbar.set_label('Aerosol Backscatter')
# plt.xlabel('Datetime (UTC)')
# plt.ylabel('Altitude (m AGL)')
# plt.title("20210517_Catonsville-MD_CHM160112_000.nc")
# plt.show()

#%% TESTBED
# Peak Picking Function

# @jit(nopython=True)
def pick_peaks(data,peak_data,PBL_index0, units=1):
    len_peaks = peak_data.shape[0]
    selected_peaks = np.zeros(len_peaks)
    selected_peaks[:] = -888
    tt = data["time"]
    zz = data["range"]*units
    
    for i in np.arange(len_peaks):
    # for i in np.arange(100):
        if i==0:
            selected_peaks[i] = PBL_index0
            i_m1 = i
        else:
            dt = tt[i]-tt[i_m1]
            if dt<=np.timedelta64(30,'m'):
                # test peak 1
                if (zz[peak_data[i,0]]-zz[selected_peaks[i_m1].astype(int)])<(200*units) and peak_data[i,0]!=0:
                    selected_peaks[i] = peak_data[i,0].astype(int)
                    i_m1 = i
                    continue
                elif (zz[peak_data[i,1]]-zz[selected_peaks[i_m1].astype(int)])<(200*units) and peak_data[i,1]!=0:
                    selected_peaks[i] = peak_data[i,1].astype(int)
                    i_m1 = i
                    continue
                elif (zz[peak_data[i,2]]-zz[selected_peaks[i_m1].astype(int)])<(200*units) and peak_data[i,2]!=0:
                    selected_peaks[i] = peak_data[i,2].astype(int)
                    i_m1 = i
                    continue
                elif (zz[peak_data[i,3]]-zz[selected_peaks[i_m1].astype(int)])<(200*units) and peak_data[i,3]!=0:
                    selected_peaks[i] = peak_data[i,3].astype(int)
                    i_m1 = i
                    continue
                else:
                    selected_peaks[i] = np.int64(-999)
                    continue
            if dt>np.timedelta64(10,'m'):
                print('No viable continuous peaks found within 10 mins.',\
                      'Peak identification stopped at time:',tt[i].astype(str)[11:19],\
                      ',index:',i)
                break
                
    return selected_peaks.astype(int)


for key in keys:
    data[key]["selected_peaks"] = pick_peaks(data[key],peaks_bottom,132)
    
    fig, ax = plt.subplots()
    ax.pcolormesh(data[key]["time"], data[key]["range"], data[key]["beta_raw"])
    # ax.plot(data[key]["selected_peaks"])