# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from datas.windprofilers.RWP import RWP as RWP

dir_path = "../sample_data/RWP"

rwp = RWP()
data = rwp.read_RWP(dir_path, LT=-4)

for key in data.keys():
    rwp.plot(data[key])
