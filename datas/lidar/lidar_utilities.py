# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 10:21:21 2022

@author: Maurice Roots

Lidar Utilities

Originally Written in Matlab by Ruben Delgado
Translated to Python by Noah Sienkiewicz
Editted and Implemented by Maurice Roots

"""
# Data Rangling
from scipy.interpolate import interp1d
import pandas as pd
import numpy as np

#%% Function Space

def SA76(zkm):
    # % [P,T,numberDensity] = SA76(zkm)
    # % Returns the pressure [Pa], T [K], numberDensity [m^-3] for the
    # % Standard Atmosphere 1976 for 0 <= zkm <= 86 km (Geometric).
    # %
    M = 28.9644 # Average molecular weight for air
    g0 = 9.80665 # m/s^2 acceleration due to gravity
    RE = 6378.14 # Earth's radius [km]
    T0 = 288.15 # 15C
    P0 = 101325.0 # 1 atmospere [Pa]
    R = 8.31447 # Gas Constant [J/K/mol]
    kB = 1.38065e-23 # [J/K] Boltzmann's Constant

    try:
        n = len(zkm)
    except TypeError:
        n = 1
        zkm = np.array([zkm])
    P = np.zeros_like(zkm)
    T = np.zeros_like(zkm)
    numberDensity =  np.zeros_like(zkm)
    ##% Geopotential Heights
    hTbl= np.array([0.0, 11.0, 20.0, 32.0, 47.0, 51.0, 71.0, RE*86.0/(RE+86.0)])
    # %Temperature gradient in each Layer
    dtdhTbl= np.array([-6.5, 0.0, 1.0, 2.8, 0.0, -2.8, -2.0])
    # %Temperature Table
    tempTbl = np.array([288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65, 186.938])
    # %Pressure Table
    pressureTbl = np.array([101325.0, 22632.7, 5475.18, 868.094, 110.92, 66.9478, 3.95715, 0.373207])


    for k in range(n):
        h = zkm[k]*RE/(zkm[k]+RE)
        # %Find the layer
        # i = int(hTbl[hTbl<= h].sum())
        i = int(np.sum(hTbl<= h))-1
        T[k] = tempTbl[i]+dtdhTbl[i]*(h-hTbl[i])
        if abs(dtdhTbl[i]) <=0.001:
            ratio = np.exp(-M*g0*(h-hTbl[i])/(R*tempTbl[i]))
        else:
            ratio = ((tempTbl[i]+dtdhTbl[i]*(h-hTbl[i]))/tempTbl[i])**(-M*g0/(R*dtdhTbl[i]))

        P[k]= ratio*pressureTbl[i];

    # numberDensity = P/(kB*T);
    return P,T

def calc_beta_rayleigh(wavelength, P, T, nanometers=True, hPa=True, celsius=True):
    if nanometers is True: wavelength = wavelength * 10**-9
    if hPa is True: P *= 100
    if celsius is True: T += 273.15
    beta_rayleigh = 2.938e-32 * (P/T) * (wavelength**(-4.0117))
    return beta_rayleigh

def calc_number_density(pressure, temperature, celsius=True, hPa=True):
    """Calculate Number Density

        Input
            pressure        -> array-like, profile of pressure
            temperature     -> array-like, profile of temperature
            celsius         -> True, Falce: option for Kelvin or Celius tempurature

        Output
            number desnsity -> array-like, profile of number density (molecules / m^-3)
    """

    kB = 1.38064852e-23 # (m2 kg s-2 K-1)
    if celsius is True: temperature += 273.15
    ND = pressure/(kB * temperature)
    return ND

def calc_index_refraction(wavelength):
    """the index of refraction of dry air at STP for wavelength lambda in nm"""
    ior = 1.0 + (5791817.0/(238.0185 - (10**6)/wavelength**2)+167909.0/(57.362-(10**6)/wavelength**2))*1e-8
    return ior

def calc_depol_ratio(wavelength):
    """depolarization ratio of gases as a function of the wavelength lambda in nm"""

    wavelengths = np.array([200.,  205.,  210.,  215.,  220.,
                  225.,  230.,  240.,  250.,  260.,
                  270.,  280.,  290.,  300.,  310.,
                  320.,  330.,  340.,  350.,  360.,
                  370.,  380.,  390.,  400.,  450.,
                  500.,  550.,  600.,  650.,  700.,
                  800.,  850.,  900.,  950.,  1000.,
                  1064.])

    depolarize = np.array([0.0454545,  0.0438372,  0.0422133,  0.0411272,  0.0400381,
                  0.0389462,  0.0378513,  0.0367534,  0.0356527,  0.0345489,
                  0.033996,   0.0328878,  0.0323326,  0.0317766,  0.0317766,
                  0.0312199,  0.0306624,  0.0306624,  0.0301042,  0.0301042,
                  0.0301042,  0.0295452,  0.0295452,  0.0295452,  0.0289855,
                  0.028425,   0.028425,  0.0278638,  0.0278638,  0.0278638,
                  0.0273018,  0.0273018,  0.0273018,  0.0273018,  0.0273018,
                  0.0273018])

    depol = interp1d(wavelengths,depolarize,kind='linear')(wavelength)
    return depol

def calc_rayleigh_scat_cross(wavelength):
    """ Calculates the Rayleigh scattering cross section per molecule [m^-3] for lambda in nm.
    """
    nstp = 2.54691e25
    rs = (1e36*24*np.pi**3*(calc_index_refraction(wavelength)**2-1)**2)
    rs /= (wavelength**4*nstp**2*(calc_index_refraction(wavelength)**2+2)**2)
    rs *= ((6+3*calc_depol_ratio(wavelength))/(6-7*calc_depol_ratio(wavelength)))
    return rs

def calc_rayleigh_extinction(wavelength, numberDensity):
    """This calculates the Rayleigh extinction coefficient in [km^-1]

        Input
            numberDensity       -> array-like, number density profile (molecules / m^-3)
            wavelength          -> float, wavelength (nm)

        Output
            rayleigh extintion  -> rayleigh extinction coefficient (km^-1)
    """
    rayleighExtinction = numberDensity*calc_rayleigh_scat_cross(wavelength)*1000
    return rayleighExtinction

def calc_rayleigh_trans(rayleigh_extinction, altitude_profile, kilometers=True):
    """Calculates the Rayleigh Transmission Profile
    """
    if kilometers is True: altitude_profile = altitude_profile / 1000
    int_alpha = [rayleigh_extinction * (altitude_profile[i] - altitude_profile[i-1]) for i in np.arange(1, len(altitude_profile))]
    # print(int_alpha)
    rayleigh_trans = np.exp(-2 * np.sum(int_alpha))
    return rayleigh_trans

def calc_rayleigh_beta_dot_trans(wavelength, pressure, temperature, altitude, nanometers=True, kilometers=True, celsius=True):
    beta = calc_beta_rayleigh(wavelength, pressure, temperature, nanometers=nanometers)
    numberDensity = calc_number_density(pressure, temperature, celsius=celsius)
    alpha = calc_rayleigh_extinction(wavelength, numberDensity)
    rayleigh_trans = calc_rayleigh_trans(alpha, altitude, kilometers=kilometers)
    beta_dot_trans = beta * (rayleigh_trans**2)
    return {"beta_rayleigh":beta, "ND":numberDensity, "alpha_rayleigh":alpha, "trans_rayleigh":rayleigh_trans, "beta_dot_trans":beta_dot_trans}


def binned_alts(data_array, altitude, bins=np.arange(0, 15000, 100)):
        data = pd.DataFrame({"data":data_array, "altitude":altitude})
        data["Alt_Bins"] = pd.cut(altitude, bins=bins)
        new = data.groupby("Alt_Bins").mean().reset_index()
        return new


#%%
if __name__ == "__main__":

    import pandas as pd

    # names=["PRES", "HGHT", "TEMP", "DWPT", "RELH", "MIXR", "DRCT", "SKNT", "THTA", "THTE", "THTV"]
    # sonde = pd.read_csv(r"C:/Users/meroo/OneDrive - UMBC/Class/Class 2022/PHYS 650/lidar/data/IAD_20200308_0Z.txt", skiprows=5,
    #                     nrows=108-5, sep="\s+", names=names)

    # kB = 1.38064852e-23 # (m2 kg s-2 K-1)
    # sonde["ND"] = sonde["PRES"]/(kB*(sonde["TEMP"]+273.15))

    # names=["PRES", "HGHT", "TEMP", "DWPT", "RELH", "MIXR", "DRCT", "SKNT", "THTA", "THTE", "THTV"]

    # sondePath = r"C:/Users/meroo/OneDrive - UMBC/Class/Class 2022/PHYS 650/lidar/data/IAD_20200308_0Z.txt"

    # sonde = pd.read_csv(sondePath, skiprows=5, nrows=108-5, sep="\s+", names=names)

    # wavelength = 1064; pressure = sonde["PRES"]; temperature=sonde["TEMP"]; altitude=sonde["HGHT"]

#%%

# Needed Packages
from datas.lidar import ceilometer
import datas.lidar.lidar_utilities as lidar_utilities
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

figPath = r"C:\Users\Magnolia\OneDrive - UMBC\Class\Class 2022\Figures"
dataPath = [r"C:\Users\Magnolia\OneDrive - UMBC\Class\Class 2022\PHYS 650\lidar\data\20200308_Catonsville-MD_CHM160112_000.nc"]

# Unpacking the Ceilometer Data from the NetCDF file
UMBC_ceilometer, files = ceilometer.importing_ceilometer(dataPath)
RCS = UMBC_ceilometer["20200308_Catonsville-MD_CHM160112_000.nc"]["beta_raw"]
altitude = UMBC_ceilometer["20200308_Catonsville-MD_CHM160112_000.nc"]["range"]
datetime = UMBC_ceilometer["20200308_Catonsville-MD_CHM160112_000.nc"]["datetime"]

# Plotting the Ceilometer curtain with attributes
parms = {"data": UMBC_ceilometer,
          "ylims": [0, 5],
          "yticks": np.arange(0.5, 5.1, 0.5),
          "title": r"UMBC Lufft CHM15K",
          "savefig": f"{figPath}\\UMBC_Ceilometer_20200508.png"}

# ceilometer.plot(**parms)


#%%

wavelength = 1064;
pressure, temperature = SA76();
rayleigh_beta_dot_trans = calc_rayleigh_beta_dot_trans(wavelength, pressure, temperature, altitude)

#%%

plt.figure(figsize=(5, 8))
# plt.plot(pressure, altitude, label="SA76: Pressure (hPa)")
plt.plot(temperature, altitude, label="SA76: Temperature (K)")
plt.legend()

#%%
print(datetime[600], datetime[900])

#%%

RCS_avg = np.mean(UMBC_ceilometer["20200308_Catonsville-MD_CHM160112_000.nc"]["beta_raw"][:, 600:900], axis=1)

beta_trans = binned_alts(beta_dot_trans, altitude*1000, bins=np.arange(3500, 6000, 20))

beta_raw_avg = binned_alts(RCS_avg, UMBC_ceilometer["20200308_Catonsville-MD_CHM160112_000.nc"]["range"], bins=np.arange(3500, 6000, 20))

#%%
plt.figure()
plt.plot(np.abs(beta_trans["data"]), beta_raw_avg["data"] / (1000**2),  "ok")
# plt.ylim(0, 0.05)
# plt.xlim(5e-8, 7e-8)


#%%

X, Y = (np.abs(beta_trans["data"].values.reshape(-1, 1)), beta_raw_avg["data"].values.reshape(-1,1))


#%%
from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(X, Y)
print(reg.score(X, Y), reg.coef_, reg.intercept_)


#%%
from scipy import stats
res = stats.linregress(np.abs(beta_trans["data"]), beta_raw_avg["data"])
print(res.rvalue, res.intercept, res.slope)


#%%
attenuated_backscatter = RCS_avg / res.slope
plt.plot(attenuated_backscatter, UMBC_ceilometer["20200308_Catonsville-MD_CHM160112_000.nc"]["range"])
plt.plot(np.abs(beta_dot_trans), altitude*1000, "k")
# plt.ylim(0, 5000)
# plt.xlim(0, 0.05)