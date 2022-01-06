# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 12:38:12 2021

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


#%% Function Space

def read_nc(dir_path):

    uL = {}

    if os.path.isdir(dir_path) is False:
        print('PATH IS NOT FOUND ON MACHINE')
        return

    print(f'Importing lidar data from {dir_path} \n')

    for filename in glob(os.path.join(dir_path, '*.nc')):
        uL[filename.split('\\')[-1]] = xr.open_dataset(filename)
        print(filename.split('\\')[-1], "-> Imported")

    return uL, list(uL.keys())

def variables(dataset, files):
    def grab_it(data, file):
        t = data[file].time
        t = t.values

        alt = data[file].range
        alt = alt.values

        # Gathering the bachscatter (i.e. beta_raw)
        r = data[file].beta_raw
        r = r.values

        # Flip the image
        r = np.array(r).T

        # Filter for negative values
        # r[r < 0] = np.nan
        r[r < 0] = 0
        np.warnings.filterwarnings('ignore')

        ''' Merge Days (couple days) '''

        alt1 = np.min(alt)
        alt2 = np.max(alt)

        x_lims = mdates.date2num(t)
        return x_lims, alt, r, alt1, alt2, t

    # if type(sav_path) is not str:
    #     sav_path = r'C:\Users\meroo\OneDrive - UMBC\Research\Figures\Preliminary'
    i = 0
    t = []
    alt = []
    r = []
    alt1 = []
    alt2 = []

    for file in files:
        uL_t, uL_alt, uL_r, uL_alt1, uL_alt2, uL_time = grab_it(dataset, file)

        if i == 0:
            t, alt, r, alt1, alt2, time = grab_it(dataset, file)

        i += 1

        if i > 1:
            t = np.append(t,uL_t)
            alt = np.append(alt, uL_alt)
            r = np.hstack((r, uL_r))
            alt1 = np.append(alt1, uL_alt1)
            alt2 = np.append(alt2, uL_alt2)
            time = np.append(time,uL_time)
    return {"x":t, "alt":alt, "backscatter":r, "time":time}


# Calculate Haar Wavelet at only one altitude 'b'
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
def pblh_timeseries(backscatter, alts, a):
    # print(f"Lengths \n backscatter: {backscatter.shape} \n alts: {alts.shape}")
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

path = r"C:\Users\magnolia\OneDrive - UMBC\Research\Code\Python\modules\incomplete\HaarCWT\test_data"
dataset, files = read_nc(path)
data = variables(dataset, files)


#%% Testbed
start_time = time.time()


a = 5
alt_stop = np.where(data["alt"]>=8000)[0][0]
alt_start = np.where(data["alt"]>=200)[0][0]


smoothed = savgol(data["backscatter"][alt_start:alt_stop, :])

#%%

a = [1, 10]
a_vals = np.arange(a[0], a[1])
# pblh = np.zeros((data["backscatter"][alt_start:alt_stop, :]).shape)
c = np.zeros((len(data["alt"][alt_start:alt_stop]), len(a_vals)))
alts = data["alt"][alt_start:alt_stop]
c_avg = np.zeros((data["backscatter"][alt_start:alt_stop, :]).shape)

for i in range(data["backscatter"][alt_start:alt_stop, :].shape[1]):
    for a, j in zip(a_vals, range(0, len(c)+1)):
        c[:,j]= covariance_profile(data["backscatter"][alt_start:alt_stop, i], a, alts)
        print(f"index: {i}, a = {a}")
    c_avg[:, i] = np.mean(c, axis=1)

end_time = time.time()
print(f"Execuded in {round(end_time - start_time, 3)} seconds")

#%% Finding Peaks

test_spot = 5000
from scipy.signal import find_peaks
alt = data["alt"][alt_start:alt_stop]
x = c_avg[:, test_spot]

peaks_top, properties = find_peaks(x, height=0, threshold=500, distance=50, prominence=(5e3, 5e6))
peaks_bottom, properties = find_peaks(-1*x, height=0, threshold=100, distance=50, prominence=(5e3, 5e6))

plt.figure()
plt.plot(x, alt)
plt.plot(x[peaks_top], alt[peaks_top], '^', color="tab:red")
plt.plot(x[peaks_bottom], alt[peaks_bottom], 'v', color="tab:red")
plt.xlim(-1e6, 1e7)
plt.ylabel("Altitude")
plt.title(f"20210517_Catonsville-MD_CHM160112_000.nc: t={test_spot}")
plt.show()

t_top = np.full(shape=alt[peaks_top].shape, fill_value=data["time"][test_spot])
t_bottom = np.full(shape=alt[peaks_bottom].shape, fill_value=data["time"][test_spot])

plt.figure(figsize=(12, 6), constrained_layout=True)
im = plt.pcolormesh(data["time"],data["alt"],data["backscatter"],cmap='jet', norm=LogNorm())
plt.axvline(data["time"][test_spot+30], linestyle='--', color="k")
plt.axvline(data["time"][test_spot-30], linestyle='--', color="k")
plt.plot(t_top, alt[peaks_top], 'v', color="pink")
plt.plot(t_bottom, alt[peaks_bottom], '^', color="pink")
plt.ylim([0,6000])
cbar = plt.colorbar()
im.set_clim(vmin=10**3.5, vmax=10**8.5)
cbar.set_label('Aerosol Backscatter')
plt.xlabel('Datetime (UTC)')
plt.ylabel('Altitude (m AGL)')
plt.title("20210517_Catonsville-MD_CHM160112_000.nc")
plt.show()

#%% PLotting Pblh

peaks_bottom = np.zeros((len(c_avg[0,:]), 50))
for i in range(0, len(c_avg[0,:])):
    peaks, properties = find_peaks(-1*c_avg[:, i], height=0, threshold=500, distance=50, prominence=(5e3, 5e4))
    order_test = sorted(((value, index) for index, value in enumerate(properties["prominences"])), reverse=True)
    print(peaks)
    for j in range(0, len(peaks)):
        peaks_bottom[i, j] = peaks[order_test[j][1]]
        print(peaks[j])
    print(i)
peaks_bottom = peaks_bottom.astype(int)

#%%

df = pd.read_csv(r"C:/Users/Magnolia/OneDrive - UMBC/Research/Code/Python/FromKylie/New folder/vanessa_lufft_pbl.csv", header=None)

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

# order_test = sorted(((value, index) for index, value in enumerate(properties["prominences"])), reverse=True)

# print(data["alt"][test[order_test[5][1]]])
#through this into the

# for i in range(data["backscatter"][alt_start:alt_stop, :].shape[1]):
#     for j in range(0, len(c)+1):
#         c[:,j]= [covariance_profile(data["backscatter"][alt_start:alt_stop, i], a, alts) for a in a_vals]
#         print(f"index: {i}, a = {a}")
#     c_avg[:, i] = np.mean(c, axis=1)