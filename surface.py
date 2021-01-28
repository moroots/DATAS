# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 16:56:52 2020

@author: Magnolia

Surface Data

"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import matplotlib.dates as mdates

class airdata:
    def byState(df, str_state):
        grouped = df.groupby("State Name")
        return grouped.get_group(str_state)

    def import_hourly(path):
        uL = {}
        print(f'Importing EPA AirData from {path} \n')
        for filename in glob.glob(os.path.join(path, '*.csv')):
           with open(os.path.join(path, filename), 'r') as f: # open in readonly mode
               names = {}
               uL2 = pd.read_csv(f, low_memory=False)
               states = unique(uL2['State Name'])
               for state in states:
                   names[state] = airdata.byState(uL2, state)
               uL[filename.split('\\')[-1]] = names
               print(filename.split('\\')[-1], '\t -> Loaded')

        name = list(uL.keys())
        return uL, name

    def format_dates(datelist, timelist):
        return [datetime(int(datelist[i].split("-")[0]), int(datelist[i].split("-")[1]), int(datelist[i].split("-")[2]), int(timelist[i].split(":")[0])) for i in range(0, len(datelist))]

    def filter_avg(in_list, in_times, start_time, end_time):
        filtered = []
        for i in range(0, len(in_list)):
            if in_times[i].hour >= start_time and in_times[i].hour <= end_time:
                filtered.append(in_list[i])
        return sum(filtered) / len(filtered)

    def filter_avg_by_day(in_list, in_times, start_hour, end_hour):
        dates = {}
        # For each item in list, filter by start/end hour and add them to a dict where the key is the day
        for i in range(0, len(in_list)):
            if in_times[i].hour >= start_hour and in_times[i].hour <= end_hour:
                date = f'{in_times[i].year}-{in_times[i].month}-{in_times[i].day}'
                if date not in dates:
                    dates[date] = [in_list[i]]
                else:
                    dates[date].append(in_list[i])
        # Calculate averages and replace individual values
        for key in dates:
            dates[key] = sum(dates[key])/len(dates[key])
        return dates

