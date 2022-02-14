# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 15:43:34 2022

@author: NoahSienkiewicz
"""
# plt.rcParams.update({'font.size': 18})
# plt.rcParams.update({'figure.figsize':[16, 9]})

import os
import numpy as np
import matplotlib.pyplot as plt
#import netCDF4 as nc

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
    
    numberDensity = P/(kB*T);
    return P,T,numberDensity

def rhon(wvl):
    from scipy.interpolate import interp1d
    # %
    # % rhon(lambda) = depolarization ratio of gases as a function of the
    # %                wavelength lambda in nm
    # %
    # %Point added at 1064
    wavelength = np.array([200.,  205.,  210.,  215.,  220.,
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
    depol = interp1d(wavelength,depolarize,kind='linear')(wvl)
    return depol

def Pchandra(wvl,theta):
    # pc = Pchandra(lambda,theta)
    # This function calculates the phase function for atmospheric
    #% scattering with the correction due to Chandrasekhar.
    #% Wavelength lambda is in nm, scattering angle theta is in radians.
    gamma = rhon(wvl)/(2.0-rhon(wvl));
    pc = 3*((1+3.0*gamma)+((1.0-gamma)*np.cos(theta)**2))/(4.0*(1.0+2.0*gamma))
    return pc

def ns(wvl):
    # function ior = ns(lambda)
    # % ns(lambda) = the index of refraction of dry air at STP 
    # %              for wavelength lambda in nm
    # %
    ior = 1.0 + (5791817.0/(238.0185 - (10**6)/wvl**2)+167909.0/(57.362-(10**6)/wvl**2))*1e-8
    return ior

def rayleigh(wvl):
    # function rs = rayleigh(lambda)
    # % rayleigh(lambda) = The Rayleigh scattering cross section per molecule [m^-3] for
    # %                 lambda in nm.
    # %
    nstp = 2.54691e25
    rs = (1e36*24*np.pi**3*(ns(wvl)**2-1)**2)/(wvl**4*nstp**2*(ns(wvl)**2+2)**2)*((6+3*rhon(wvl))/(6-7*rhon(wvl)))
    return rs

def betamol(wvl,zkm):
    # bm = betamol(lambda,zkm)
    # This calculates the molecular (Rayleigh) backscattering coefficient
    # in [km^-1 sr^-1] with the assumption of a SA76 atmosphere.
    # lambda is the wavelength in nm, and zkm is the altitude in km.
    P,T,numberDensity = SA76(zkm)
    bm = 1000*numberDensity*Pchandra(wvl,np.pi)*rayleigh(wvl)/(4*np.pi);
    return bm

def alphamol(wvl, zkm):
    # function am = alphamol(lambda,zkm)
    # % am = alphamol(lambda,zkm)
    # % This calculates the molecular (Rayleigh) extinction coefficient
    # % in [km^-1] with the assumption of a SA76 atmosphere.
    # % lambda is the wavelength in nm, and zkm is the altitude in km.
    P,T,n = SA76(zkm)
    am = n*rayleigh(wvl)*1000
    return am

def rayleighOT(wvl,z0,z1):
# function tau=rayleighOT(lambda,z0,z1)
# % tau is the Rayleigh optical thickness [1]
# % between z0 and z1 [km] at wavelength
# % lambda [nm]
    try:
        n = len(z1)
    except TypeError:
        n = 1
        z1 = np.array([z1])
        z0 = np.array([z0])
    tau= np.zeros((1,n))
    for i in range(n):
        dz=(z1[i]-z0)/1000
        zkm = np.arange(z0,z1[i]+dz,dz)
        alpha= alphamol(wvl,zkm)
        tau[i]=np.sum(alpha)*dz
    return tau