# -*- coding: utf-8 -*-
"""
Created on Thu May  4 09:38:27 2023

@author: meroo
"""


import numpy as np
import pandas as pd
import matplotlib as plt
from pathlib import Path
import copy
from tqdm import tqdm

def debug(func):
    def wrapper(*args, **kwargs):
        print('*' * 80)
        print(f'Calling function {func.__name__}\nargs={args}\nkwargs={kwargs}')
        result = func(*args, **kwargs)
        print(f'Return of function {func.__name__} is {result}')
        print('*' * 80 + '\n')
        return result
    return wrapper


class Surface_EPA:
    def __init__(self):
        return
    
    @staticmethod
    def import_EPA_zip_single(filename, **kwargs):
        return pd.read_csv(filename, compression="zip", low_memory=False)

    @classmethod
    def EPA_zip_to_parquet(cls, filename, **kwargs):
        data = cls.import_EPA_zip_single(filename, **kwargs)
        data.to_parquet(path=(str(filename) + ".gzip"), compression='gzip')
        return
    
    @staticmethod
    def read_EPA_parquet(filename):
        cols = ["Site Num", "Sample Measurement", "State Name", 'Latitude', 'Longitude', "Date GMT", 'Time GMT']
        return pd.read_parquet((str(filename) + ".gzip"), engine="fastparquet", columns=cols)
    
    def import_EPA_parquets(self, filenames, loading=True):
        self.data = {}; self.sites = {}
        self.names = [filename.name for filename in filenames]
        
        if loading: 
            print(f"Importing EPA Parquet Files: {filenames[0].parent}")
            filenames = tqdm(filenames)
            
        for filename in filenames:
            self.data[filename.name] = self.read_EPA_parquet(filename)
            self.sites[filename.name] = self.data[filename.name].drop(columns=self.data[filename.name].columns.difference(['Site Num', 'Latitude', 'Longitude'])).drop_duplicates()
        return self
    
    @staticmethod
    def select_state(df, state_name='Maryland', **kwargs):
        try: 
            # Grab only the site within the given state
            state = copy.deepcopy(df[df["State Name"] == state_name])

            # Grab the Site Number and Location of Monitoring sites in 
            state_sites = state.drop(columns=state.columns.difference(['Site Num', 'Latitude', 'Longitude'])).drop_duplicates()

            # Creating a timestamp column in GMT
            state["Timestamp GMT"] = pd.to_datetime((state["Date GMT"] + ' ' + state['Time GMT']), utc=True)

            # Create a Datetime and Site Number index for the series (MultiIndexed)
            state.set_index(["Site Num", "Timestamp GMT"], inplace=True)

            # Only keep the Sample Measurment and Uncertainty
            state.drop(columns=state.columns.difference(['Sample Measurement', 'Uncertainty']), inplace=True)
        except: 
            pass

        return state, state_sites
    
    def select_states(self, state_name="Maryland", loading=True, **kwargs):
        names = self.names
        
        if loading: 
            print(f"Selecting Only: {state_name}")
            names = tqdm(names)
            
        for name in names:
            self.data[name], self.sites[name] = self.select_state(self.data[name], state_name, **kwargs)
            
        return self
    
    @staticmethod
    def diurnal_nocturnal_mean(state, state_sites, loading=True):
        """
        Compute diurnal and nocturnal means, standard deviations, maximums, and minimums for a given state's air quality
        data, using EPA surface monitor data with hourly measurements.

        Args:
            state (pd.DataFrame): DataFrame containing hourly air quality data for a given state. Must have a MultiIndex
                                  with Site ID and Date as the first two levels.
            state_sites (pd.DataFrame): DataFrame containing site information for the state. Must include a column named
                                        "Site Num" containing the Site ID values used in the state DataFrame.

        Returns:
            dict: A dictionary containing DataFrames with the computed diurnal and nocturnal statistics for each Site ID.
                  Each DataFrame has columns for Diurnal Mean, Nocturnal Mean, Diurnal STD, Nocturnal STD, Diurnal Max, and
                  Nocturnal Min, with dates as the index.
        """
        data = {}
        state.sort_index(inplace=True)
        
        sites = state_sites["Site Num"]
        
        if loading:
            print("Calculating Diurnal and Nocturnal Averages by Sites")
            sites = tqdm(sites)
            
        for site in sites:
            # Select data between 10am-3pm for diurnal mean
            diurnal = state.loc[(site, ), "Sample Measurement"].between_time("10:00", "15:00")
            # Select data between 8pm-3am for nocturnal mean
            nocturnal = state.loc[(site, ), "Sample Measurement"].between_time("20:00", "03:00")
            # Compute diurnal and nocturnal means, standard deviations, maximums, and minimums using resampling
            data[str(site)] = pd.DataFrame({
                "Diurnal Mean": diurnal.resample("D").mean(),
                "Nocturnal Mean": nocturnal.resample("D").mean(),
                "Diurnal STD": diurnal.resample("D").std(),
                "Nocturnal STD": nocturnal.resample("D").std(),
                "Diurnal Max": diurnal.resample("D").max(),
                "Nocturnal Min": nocturnal.resample("D").min()
            })
        return data
    
    def diurnal_nocturnal_means(self, rm_nan=True, loading=True):
        self.diurnal = {}; names = self.names
        
        if loading: 
            print(f"Creating Diurnal and Nocturnal Means")
            names = tqdm(names)
            
        for name in names:
            data = self.diurnal_nocturnal_mean(self.data[name], self.sites[name], loading=False)
            self.diurnal[name] = pd.concat(data, axis=1, join="inner")
            if rm_nan: self.diurnal[name].dropna(axis=1, how='any', inplace=True)
            
        return self
    
    
if __name__ == "__main__":
    data = Surface_EPA()