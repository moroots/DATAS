# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 11:17:31 2021

@author: Magnolia
"""

# Import Module
from ftplib import FTP

# Fill Required Information
HOSTNAME = "madis-data.ncep.noaa.gov"
# USERNAME = "dlpuser@dlptest.com"
# PASSWORD = "eUj8GeW55SvYaswqUyDSm5v6N"

# Connect FTP Server
ftp = FTP(HOSTNAME)
ftp.login()

# # force UTF-8 encoding
# ftp_server.encoding = "utf-8"


# Get list of files
ftp.dir()

ftp.cwd("/LDAD/profiler/netCDF/")
ftp.dir()
filename = "20211209_0100.gz"
save_path = r"C:/Users/Magnolia/OneDrive - UMBC\Research/Data/Wind Profilier/"
save_name = save_path+filename
perm_name = save_path+filename.replace(".gz", ".nc")
ftp.retrbinary("RETR " + filename, open(save_name, 'wb').write)
ftp.quit()

import gzip
import shutil
with gzip.open(save_name, 'rb') as f_in:
    with open(perm_name, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

#%%
import netCDF4 as nc
fn = perm_name
ds = nc.Dataset(fn)
# Dataset = ds.__dict__
var = ds.variables
var.keys()

