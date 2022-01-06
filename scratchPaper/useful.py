# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 16:55:15 2020

@author: Magnolia

Useful Functions

"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import matplotlib.dates as mdates

def truncate(n1, n2):
    return int(n1 * (10**n2)) / (10**n2)

def unique(list1):
    x = np.array(list1)
    return np.unique(x)

def fit_it(x, y, degree):
    import numpy as np
    xp = np.arange(min(y), max(y), .1)
    z = np.polyfit(x,y,degree)
    p = np.poly1d(z)
    return xp, p(xp), z

def string_in_txt(file_name, string_to_search):
    """Search for the given string in file and return lines containing that string,
    along with line numbers"""
    line_number = 0
    list_of_results = []
    # Open the file in read only mode
    with open(file_name, 'r') as read_obj:
        # Read all lines in the file one by one
        for line in read_obj:
            # For each line, check if line contains the string
            line_number += 1
            if string_to_search in line:
                # If yes, then add the line number & line as a tuple in the list
                list_of_results.append((line_number, line.rstrip()))

    # Return list of tuples containing line numbers and lines where string is found
    return list_of_results

class paths:
    def __init__(self, machine):
        if machine == 'iThink':
            p = r'C:\Users\meroo'
        if machine == 'Magnolia':
            p = r'C:\Users\Magnolia'
        self.pan = p + r'\OneDrive - UMBC\Research\Data\Pandora\Pandonia\Use'
        self.air = p + r'\OneDrive - UMBC\Research\Data\AirNow\USE'
        self.fig = p + r'\OneDrive - UMBC\Research\Figures\Preliminary'
        self.sav = p + r'\OneDrive - UMBC\Research\Data\Processed'
        self.air = p + r'\OneDrive - UMBC\Research\Data\AirNow\USE'
        self.current = os.getcwd()

class WaterVapMix:
    def __init__(self, temp, rh, press):
        temp = np.array(temp); temp = temp.astype(np.float)
        rh = np.array(rh); rh = rh.astype(np.float);
        press = np.array(press); press = press.astype(np.float)
        self.WexlerSat = np.exp(-2.9912729 * 10**3 * temp**-2 - 6.0170128 * 10**3 * temp**-1 + 1.887643854 * 10**1 -
                          2.8354721 * 10**-2 * temp + 1.7838301 * 10**-5 * temp**2 - 8.4150417 * 10**-10 * temp**3 +
                          4.4412543 * 10**-13 * temp**4 + 2.858487 * np.log(temp)) * 10 / 1000

        # Mixing Ratio
        self.Wexler = 10 * rh * 0.62197 * (self.WexlerSat / (press - (rh/100) * self.WexlerSat))