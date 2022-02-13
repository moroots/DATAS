# -*- coding: utf-8 -*-
"""
Created on Sat Feb 12 18:18:06 2022

@author: Maurice Roots

"""

with open("requirements.txt", "r") as f:
    reqs = f.read()

reqs = reqs.replace(" ", "")
reqs = reqs.split("\n")
reqs = [ x for x in reqs if "#" not in x ]
reqs = ' '.join(reqs)
reqs = reqs.split()


