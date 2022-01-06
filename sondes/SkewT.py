# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 13:06:56 2020

@author: Magnolia
"""


from datetime import datetime
from metpy.units import units
from siphon.simplewebservice.wyoming import WyomingUpperAir

#Choosing the Day, Time, Location
date = datetime(2020, 3, 9, 0)
station = 'IAD'

#Grabbing the Data
def WyomingSondes(date_time, station_id):

    df = WyomingUpperAir.request_data(date, station)

    #Extracting the variables and attaching units
    p = df['pressure'].values * units(df.units['pressure'])
    T = df['temperature'].values * units(df.units['temperature'])
    Td = df['dewpoint'].values * units(df.units['dewpoint'])
    u = df['u_wind'].values * units(df.units['u_wind'])
    v = df['v_wind'].values * units(df.units['v_wind'])
    heights = df['height'].values * units(df.units['height'])

    return p, T, Td, u, v, heights, df

p, T, Td, u, v, heights, df = WyomingSondes(date, station)

#%% Plotting
import matplotlib.pyplot as plt
import metpy.plots as plots

import numpy as np
import metpy.calc as mpcalc

def SkewT_plot(p, T, Td, heights, u=0, v=0, wind_barb=0, p_lims=[1000, 100], T_lims=[-50,35], metpy_logo=0, plt_lfc=0, plt_lcl=0, plt_el=0, title=None):

    #plotting
    fig = plt.figure(figsize=(10, 10))
    skew = plots.SkewT(fig)

    skew.plot(p, T, 'red')  #Virtual Temperature
    skew.plot(p, Td, 'green')   #Dewpoint

    skew.ax.set_ylim(p_lims)
    skew.ax.set_xlim(T_lims)

    if wind_barb==1:
        # resampling wind barbs
        interval = np.logspace(2, 3) * units.hPa
        idx = mpcalc.resample_nn_1d(p, interval)
        skew.plot_barbs(p[idx], u[idx], v[idx])

    #Showing Adiabasts and Mixing Ratio Lines
    skew.plot_dry_adiabats()
    skew.plot_moist_adiabats()
    skew.plot_mixing_lines()

    parcel_path = mpcalc.parcel_profile(p, T[0], Td[0])
    skew.plot(p, parcel_path, color='k')

    # CAPE and CIN
    skew.shade_cape(p, T, parcel_path)
    skew.shade_cin(p, T, parcel_path)

    # Isotherms and Isobars
    # skew.ax.axhline(500 * units.hPa, color='k')
    # skew.ax.axvline(0 * units.degC, color='c')

    # LCL, LFC, EL
    lcl_p, lcl_T = mpcalc.lcl(p[0], T[0], Td[0])
    lfc_p, lfc_T = mpcalc.lfc(p, T, Td)
    el_p, el_T = mpcalc.el(p, T, Td)

    if plt_lfc==1:
        skew.ax.axhline(lfc_p, color='k')

    if plt_lcl==1:
        skew.ax.axhline(lcl_p, color='k')
        # skew.ax.text(lcl_p, )

    if plt_el==1:
        skew.ax.axhline(el_p, color='k')

    if metpy_logo==1: plots.add_metpy_logo(fig, x=55, y=50)

    decimate = 3
    for p, T, heights in zip(df['pressure'][::decimate], df['temperature'][::decimate], df['height'][::decimate]):
        if p >= 700: skew.ax.text(T+1, p, round(heights, 0))

    plt.title(title)

    plt.show()
    return

#%% CAPE and CIN

# print(mpcalc.surface_based_cape_cin(p, T, Td))  # Surface Parcel Based CAPE and CIN
# print(mpcalc.most_unstable_cape_cin(p, T, Td))  # Using the most unstable layer

# p_ML, T_ML, Td_ML = mpcalc.get_layer(p, T, Td, heights=heights, depth=3*units.km)

# SkewT_plot(p_ML, T_ML, Td_ML, heights, plt_lcl=0, p_lims=[1000, 700], T_lims=[-10, 35], title=f'IAD: {date}')

#%% Testbed

SkewT_plot(p, T, Td, heights, plt_lcl=1, plt_lfc=1, plt_el=1, p_lims=[1000, 100], T_lims=[-50, 35], title=f'IAD: {date}')