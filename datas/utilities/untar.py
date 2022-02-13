# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 16:01:04 2022

@author: meroo
"""

import tarfile
import os.path
from pathlib import Path

def untar(filePath, destination=None, disp=True):
    
    """ 
    filePath:       full path including file name
    destination:    the full path to desired directory
    disp:           short for dispplay, True will print contents
    
    """
    p = Path(filePath)

    if p.is_file():
        
        file = tarfile.open(filePath)
        
        if disp is True: print(file.getnames())
        
        if destination:
            destination = Path(destination)
            if destination.is_dir():
                file.extractall(destination)
        else: 
            destination = p.parent
            file.extractall(destination)
            
        file.close()
        
        print(f"File Extracted to: {destination}")
        
    else: 
        print("Please supply a valid file path")
    
if __name__ == "__main__":
    filePath = r"C:/Users/meroo/OneDrive - UMBC/Research/Analysis/OWLETS-2/HYSPLIT/From Loughner/writepbl.tar.gz"
    untar(filePath)