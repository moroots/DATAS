# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 18:30:32 2022

@author: meroo
"""

# %% Import Packages

# ploting
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LogNorm

# data processing
from scipy.signal import savgol_filter
from scipy.signal import find_peaks
import xarray as xr
import pandas as pd

# utilities
from pathlib import Path
from numba import jit
import numpy as np
import time

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

def pick_peaks_001(data,peak_data,apriori_height=2000, Dy=200, Dt=np.timedelta64(30,'m'), **kwargs):
    
    len_peaks = np.int64(len(peak_data))
    selected_peaks = np.ones(len_peaks)*-1
    blanks = 0
    for i in np.arange(len_peaks):
        if i==0:
            selected_peaks[i] = np.where(data["range"] >= apriori_height)[0][0]
            j = i
        else:
            dt = data["time"][i]-data["time"][j]
            if dt<=Dt:
                if (np.abs(data["range"][peak_data[i,0]]-data["range"][np.int64(selected_peaks[j])])<=(Dy)) and (peak_data[i,0]!=0):
                    selected_peaks[i] = np.int64(peak_data[i,0])
                    j = i
                    continue
                elif (np.abs(data["range"][peak_data[i,1]]-data["range"][np.int64(selected_peaks[j])])<=(Dy)) and (peak_data[i,1]!=0):
                    selected_peaks[i] = np.int64(peak_data[i,1])
                    j = i
                    continue
                elif (np.abs(data["range"][peak_data[i,2]]-data["range"][np.int64(selected_peaks[j])])<=(Dy)) and (peak_data[i,2]!=0):
                    selected_peaks[i] = np.int64(peak_data[i,2])
                    j = i
                    continue
                elif (np.abs(data["range"][peak_data[i,3]]-data["range"][np.int64(selected_peaks[j])])<=(Dy)) and (peak_data[i,3]!=0):
                    selected_peaks[i] = np.int64(peak_data[i,3])
                    j = i
                    continue
                else:
                    selected_peaks[i] = np.int64(-999)
                    continue
            else:
                blanks += 1
                if blanks < 0.1*len_peaks:
                    print("Too few points, attempt new apriori_height")
                    break
                
    return np.int64(selected_peaks)

def pick_peaks_002(data, key, peak_data, Dy=200, Dt=np.timedelta64(30,'m'), **kwargs):
    
    len_peaks = np.int64(len(peak_data))
    selected_peaks = np.ones(len_peaks)*-1
    blanks = 0
    
    file = open(r"C:\Users\meroo\Box\Roots\PBL Codes\code\log\log_test.txt", "a")
    file.write(f"{key}\n --- ")
    for i in np.arange(len_peaks):
        if i==0:
            selected_peaks[i] = peak_data[0,0]
            j = i
        else:
            dt = data["time"][i]-data["time"][j]
            if dt<=Dt:
                if (np.abs(data["range"][peak_data[i,0]]-data["range"][np.int64(selected_peaks[j])])<=(Dy)) and (peak_data[i,0]!=0):
                    selected_peaks[i] = np.int64(peak_data[i,0])
                    j = i
                    continue
                elif (np.abs(data["range"][peak_data[i,1]]-data["range"][np.int64(selected_peaks[j])])<=(Dy)) and (peak_data[i,1]!=0):
                    selected_peaks[i] = np.int64(peak_data[i,1])
                    j = i
                    continue
                elif (np.abs(data["range"][peak_data[i,2]]-data["range"][np.int64(selected_peaks[j])])<=(Dy)) and (peak_data[i,2]!=0):
                    selected_peaks[i] = np.int64(peak_data[i,2])
                    j = i
                    continue
                elif (np.abs(data["range"][peak_data[i,3]]-data["range"][np.int64(selected_peaks[j])])<=(Dy)) and (peak_data[i,3]!=0):
                    selected_peaks[i] = np.int64(peak_data[i,3])
                    j = i
                    continue
                else:
                    selected_peaks[i] = np.int64(-999)
                    continue
            else:
                blanks += 1
                if blanks < 0.1*len_peaks:
                    file.write(f"Too many blanks: index {i}\n")
                    # print("Too few points... First peak insufficient")
                    # break
    file.close()            
    return np.int64(selected_peaks)

def estimate_pblh(data, a_vals, alt_constraint=[100, 4000], **kwargs):
    
    if not "parms__peak_selection" in kwargs.keys():
        parms__peak_selection = {"apriori_height":1500, "Dy":500, "Dt":np.timedelta64(60,'m')}
            
    for key in keys:
        
        # Smoothing
        alt_stop = np.where(data[key]["range"]>=alt_constraint[1])[0][0]
        alt_start = np.where(data[key]["range"]>=alt_constraint[0])[0][0]
        smoothed = savgol(data[key]["beta_raw"][alt_start:alt_stop, :])
        
        # Covariance Transform
        c = np.zeros((len(smoothed), len(a_vals)))
        alts = data[key]["range"][alt_start:alt_stop]
        c_avg = np.zeros(smoothed.shape)
        
        for i in np.arange(smoothed[:, :].shape[1]):
            for a, j in zip(a_vals, np.arange(0, len(c)+1)):
                c[:,j]= covariance_profile(smoothed[:, i], a, alts)
            c_avg[:, i] = np.mean(c, axis=1)
    
        # Finding Peaks
        peaks_bottom = np.zeros((len(c_avg[0,:]), 50))
        for i in np.arange(0, len(c_avg[0,:])):
            peaks, properties = find_peaks(-1*c_avg[:, i], height=0, prominence=(5e3, 5e4))
            order_test = sorted(((value, index) for index, value in enumerate(properties["prominences"])), reverse=True)
            for j in range(0, len(peaks)):
                peaks_bottom[i, j] = peaks[order_test[j][1]]
        peaks_data = np.int64(peaks_bottom)
        
        # Selecting PBLH
        selected_peaks = pick_peaks_002(data[key], key, peaks_data, **parms__peak_selection)
        df = pd.DataFrame({"peaks": selected_peaks, "time": data[key]["time"]})
        df = df[df.peaks > 0]
        df["heights"] = data[key]["range"][df["peaks"]]
        
        data[key]["pblh"] = df   
        data[key]["peaks"] = peaks_data
        data[key]["c_avg"] = c_avg
        data[key]["smoothed"] = smoothed
        data[key]["selected_peaks"] = selected_peaks
        data[key]["alt_start_stop"] = [alt_start, alt_stop]
    
    return data


if __name__ == "__main__":
    
    file_start_time = time.time()
    
    FilePaths = [r"C:/Users/meroo/Box/Roots/PBL Codes/data/20210502_Catonsville-MD_CHM160112_000.nc",
r"C:/Users/meroo/Box/Roots/PBL Codes/data/20210503_Catonsville-MD_CHM160112_000.nc",
r"C:/Users/meroo/Box/Roots/PBL Codes/data/20210504_Catonsville-MD_CHM160112_000.nc"]
    
    data, files = importing_ceilometer(FilePaths)
    keys = list(data.keys())
    
    data = estimate_pblh(data=data, a_vals=np.arange(1, 500, 10))

#%%

    df = pd.read_csv(r"C:/Users/meroo/OneDrive - UMBC/Research/Code/Python/FromKylie/New folder/vanessa_lufft_pbl.csv", header=None)
    
    fig, ax1 = plt.subplots(figsize=(15, 8))
    for key in keys:
        # fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(15,8))
        
        
        im1 = ax1.pcolormesh(data[key]["time"], data[key]["range"], data[key]["beta_raw"], cmap='jet', norm=LogNorm())
        im1.set_clim(vmin=10**5, vmax=10**5.5)
        
        alt_start, alt_stop = data[key]["alt_start_stop"]
        
        # im2 = ax2.pcolormesh(data[key]["time"], data[key]["range"][alt_start:alt_stop], data[key]["c_avg"], cmap='jet', norm=LogNorm())
        # im2.set_clim(vmin=10**-3, vmax=10**2)
        
        # ax1.plot(data[key]["pblh"].time,data[key]["pblh"].heights, "^", color="k",markersize=3)
        
        # ax1.plot(data[key]["time"],df[1], "^", color="pink",markersize=3)
        
    ax1.set_ylim([200,4000])
    # ax2.set_ylim([200,4000])
    cbar = fig.colorbar(im1, ax=ax1, pad=0.01)
    # cbar = fig.colorbar(im2, ax=ax2, pad=0.01)
    cbar.set_label('Aerosol Backscatter')
    ax1.set_xlabel('Datetime (UTC)')
    ax1.set_ylabel('Altitude (m AGL)')
    plt.suptitle(key)
    plt.show()

    file_end_time = time.time()
    print(f"Execuded file in {round(file_end_time - file_start_time, 3)} seconds")
    
#%% 

 
    
    