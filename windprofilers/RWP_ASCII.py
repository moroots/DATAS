# -*- coding: utf-8 -*-
"""
Created on Sun Jan  9 18:19:31 2022

@author: meroo
"""


import pandas as pd 

    
    
fileNames = [r"C:/Users/meroo/OneDrive - UMBC/Research/Analysis/May2021/data/RWP/w21137.cns"]


for fileName in fileNames:
    # Search for the given string in lines and return the 
    data = {}
    with open(fileName, 'r') as file:
        timestep = 6 
        for position, line in enumerate(file):
            timestep += 6; 
            if "beltsville" or "$" in line:
                print(line)
                data[f"{timestep}"] = pd.read_csv(file, skiprows=8, nrows=63, sep="\s+")
                
