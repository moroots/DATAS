# -*- coding: utf-8 -*-
"""
Created on Tue May 10 15:58:05 2022

@author: NoahSienkiewicz
"""

import os
import numpy as np
import datetime as dt

AeroDT = lambda x : dt.datetime.strptime(x,'%d:%m:%Y%H:%M:%S') #helper function for time handling in AERONET file
AeroDT = np.vectorize(AeroDT) #vectorize time helper

def IsFloat(num): 
    '''
    Helper function for checking if a number is a float. Defined here so as to be
    easy to make Numpy vectorized

    Parameters
    ----------
    num : Any obj
        Object for evaluation

    Returns
    -------
    bool
        TRUE or FALSE if num is a float or not.

    '''
    try:
        np.float32(num)
        return True
    except ValueError:
        return False
IsFloat = np.vectorize(IsFloat) #vectorized

def ReadAERONET(data_dir, wvl, datestring=''):
    '''
    Function to parse AERONET data files.

    Parameters
    ----------
    data_dir : str
        Directory string which contains AERONET data files for a single site.
    wvl : str
        One of 440, 675, 870, or 1020. Wavelength of field desired in nm.
    datestring : str, optional
        The default is ''. If desire to know AERONET data which is closest to a certain
        datetime, fill this in as a str in format (YYYMMDDTHHMMSS). Functino will return index of data
        nearest

    Returns
    -------
    aeronet : dict
        aeronet data dict containing file contents.
    time_ind : int, optional
        The default is 0. If given a datestring, returns needed index for closest AERONET time.

    '''
    filenames = np.array([os.listdir(data_dir)]).flatten()
    if len(datestring)!=0:
        dtime = dt.datetime.strptime(datestring,'%Y%m%dT%H%M%S')
        
    assert(np.isin(wvl,np.array(['440','675','870','1020']))), 'Check Wavelength is one of 440, 675, 870, or 1020'
    
    aeronet = {}
    for i in filenames:
        tmp_fname = os.path.join(data_dir,i)
        if tmp_fname.split('.')[-1]=='aod':
            full_dat = np.loadtxt(tmp_fname,delimiter=',',skiprows=6,dtype='object')
            ind_fine = np.where(full_dat[0,:]==f'AOD_Extinction-Fine[{wvl}nm]')[0][0]
            ind_crse = np.where(full_dat[0,:]==f'AOD_Extinction-Coarse[{wvl}nm]')[0][0]
            ind_ang = np.where(full_dat[0,:]=='Extinction_Angstrom_Exponent_440-870nm-Total')[0][0]
            time_dat = np.loadtxt(tmp_fname,delimiter=',',skiprows=6,dtype='object',usecols=(0,1,2))
            time_arr = time_dat[1:,1:].sum(axis=1)
            time_arr = AeroDT(time_arr)
            
            aeronet['datetime'] = time_arr
            aeronet['aod_f'] = np.float32(full_dat[1:,ind_fine])
            aeronet['aod_c'] = np.float32(full_dat[1:,ind_crse])
            aeronet['ang_exp'] = np.float32(full_dat[1:,ind_ang])
            
        elif tmp_fname.split('.')[-1]=='rin':
            full_dat = np.loadtxt(tmp_fname,delimiter=',',skiprows=6,dtype='object')
            ind_real = np.where(full_dat[0,:]==f'Refractive_Index-Real_Part[{wvl}nm]')[0][0]
            ind_imag = np.where(full_dat[0,:]==f'Refractive_Index-Imaginary_Part[{wvl}nm]')[0][0]
            
            aeronet['n_real'] = np.float32(full_dat[1:,ind_real])
            aeronet['n_imag'] = np.float32(full_dat[1:,ind_imag])
            
        elif tmp_fname.split('.')[-1]=='siz':
            full_dat = np.loadtxt(tmp_fname,delimiter=',',skiprows=6,dtype='object')
            size_inds = np.where(IsFloat(full_dat[0,:]))[0]
            
            aeronet['size_bins'] = np.float32(full_dat[0,size_inds])
            aeronet['size_conc'] = np.float32(full_dat[1:,size_inds])
            
        elif tmp_fname.split('.')[-1]=='ssa':
            full_dat = np.loadtxt(tmp_fname,delimiter=',',skiprows=6,dtype='object')
            ssa_ind = np.where(full_dat[0,:]==f'Single_Scattering_Albedo[{wvl}nm]')[0][0]
            
            aeronet['ssa'] = np.float32(full_dat[1:,ssa_ind])
            
        elif tmp_fname.split('.')[-1]=='vol':
            full_dat = np.loadtxt(tmp_fname,delimiter=',',skiprows=6,dtype='object')
            ind_volc_f = np.where(full_dat[0,:]=='VolC-F')[0][0]
            ind_reff_f = np.where(full_dat[0,:]=='REff-F')[0][0]
            ind_vmr_f = np.where(full_dat[0,:]=='VMR-F')[0][0]
            ind_std_f = np.where(full_dat[0,:]=='Std-F')[0][0]
            
            ind_volc_c = np.where(full_dat[0,:]=='VolC-C')[0][0]
            ind_reff_c = np.where(full_dat[0,:]=='REff-C')[0][0]
            ind_vmr_c = np.where(full_dat[0,:]=='VMR-C')[0][0]
            ind_std_c = np.where(full_dat[0,:]=='Std-C')[0][0]
            
            aeronet['vol-conc-f'] = np.float32(full_dat[1:,ind_volc_f])
            aeronet['r-eff-f'] = np.float32(full_dat[1:,ind_reff_f])
            aeronet['vol-mean-rad-f'] = np.float32(full_dat[1:,ind_vmr_f])
            aeronet['vol-std-f'] = np.float32(full_dat[1:,ind_std_f])
            
            aeronet['vol-conc-c'] = np.float32(full_dat[1:,ind_volc_c])
            aeronet['r-eff-c'] = np.float32(full_dat[1:,ind_reff_c])
            aeronet['vol-mean-rad-c'] = np.float32(full_dat[1:,ind_vmr_c])
            aeronet['vol-std-c'] = np.float32(full_dat[1:,ind_std_c])
    
    tmp = aeronet['datetime'] - dtime
    time_ind = np.where(np.abs(tmp)==np.min(np.abs(tmp)))[0][0]
    if len(datestring) !=0:
        return aeronet, time_ind
    else:
        return aeronet
        
def ShowAERONET(aeronet, time_ind=0):
    '''
    Quick visualizer tool for AERONET data.

    Parameters
    ----------
    aeronet : dict
        Dict containing AERONET data which needs to be visualized. See ReadAERONET().
    time_ind : int, optional
        The default is 0. Index of AERONET data to visualize.

    Returns
    -------
    f : matplotlib.pyplot.figure
        Figure object of resultant plot.
    ax : matplotlib.pyplot.axis
        Axis object of resultant plot.

    '''
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 18})
    plt.rcParams.update({'figure.figsize':[16, 9]})
    
    f, ax= plt.subplots(1, 2, gridspec_kw={'width_ratios': [3, 1]})
    ax[0].plot(aeronet['size_bins'],aeronet['size_conc'][time_ind],marker='o')
    ax[0].set_xscale('log')
    ax[0].set_title('Retrieval Size Distribution')
    ax[0].set_xlabel('Particle Radius ($\\mu m$)')
    ax[0].set_ylabel('Volume Conc. ($\\mu m^3/\\mu m^2$)')
    
    ax[1].axis('off')
    top = 0.95
    step = 0.05
    ax[1].text(0,top+step,'AERONET {}-{}nm Data'.format(data_dir.split('\\')[-1][18:],wvl))
    ax[1].text(0,top-1*step,'Time: {}'.format(aeronet['datetime'][time_ind].isoformat()))
    
    ax[1].text(0,top-2*step,'    AOD_Fine: {:.3f}'.format(aeronet['aod_f'][time_ind]))
    ax[1].text(0,top-3*step,'    AOD_Coarse: {:.3f}'.format(aeronet['aod_c'][time_ind]))
    
    ax[1].text(0,top-4*step,'    Refra_Index: {:.3f}+{:.2e}$j$'.format(aeronet['n_real'][time_ind],aeronet['n_imag'][time_ind]))
    
    ax[1].text(0,top-5*step,'    Angstrom Exp: {:.3f}'.format(aeronet['ang_exp'][time_ind]))
    ax[1].text(0,top-6*step,'    Single-Scat. Albedo: {:.3f}'.format(aeronet['ssa'][time_ind]))
    
    ax[1].text(0,top-8*step,'Size Distr. Parameters')
    ax[1].text(0,top-9*step,'  Fine Mode')
    ax[1].text(0,top-10*step,'    $r_{eff}$='+'{:.3f}'.format(aeronet['r-eff-f'][time_ind]))
    ax[1].text(0,top-11*step,'    $C_{vol}$='+'{:.3f}'.format(aeronet['vol-conc-f'][time_ind]))
    ax[1].text(0,top-12*step,'    $\\ln({r_v})$='+'{:.3f}'.format(aeronet['vol-mean-rad-f'][time_ind]))
    ax[1].text(0,top-13*step,'    $\\sigma_v$='+'{:.3f}'.format(aeronet['vol-std-f'][time_ind]))
    
    ax[1].text(0,top-15*step,'  Coarse Mode')
    ax[1].text(0,top-16*step,'    $r_{eff}$='+'{:.3f}'.format(aeronet['r-eff-c'][time_ind]))
    ax[1].text(0,top-17*step,'    $C_{vol}$='+'{:.3f}'.format(aeronet['vol-conc-c'][time_ind]))
    ax[1].text(0,top-18*step,'    $\\ln({r_v})$='+'{:.3f}'.format(aeronet['vol-mean-rad-c'][time_ind]))
    ax[1].text(0,top-19*step,'    $\\sigma_v$='+'{:.3f}'.format(aeronet['vol-std-c'][time_ind]))
    plt.subplots_adjust(top=0.935,
                        bottom=0.108,
                        left=0.091,
                        right=0.824,
                        hspace=0.2,
                        wspace=0.047)
    return f, ax

#%%

if __name__=='__main__':
    
    data_dir = r"C:\Users\NoahSienkiewicz\Documents\Working\HARP_CalValidation\AeronetDatasets\20210922_080249\20210921_20210923_MCO-Hanimaadhoo"
    datestring = '20200603T124215'
    wvl = '675'
    
    aeronet,time_ind = ReadAERONET(data_dir, wvl, datestring=datestring)
    f,ax = ShowAERONET(aeronet, time_ind=time_ind)