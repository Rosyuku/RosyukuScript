#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 22:46:15 2018

@author: kazuyuki
"""

import RosyukuScript.RosyukuScript as rs
import pandas as pd
from sklearn import datasets
import importlib
importlib.reload(rs)
 
if __name__ == "__main__":
     
    df = pd.read_csv("iris.csv")   
    skData = rs.pan2sk(df, target='Name', name='iris')
    
    skData = datasets.load_iris()
    df = rs.sk2pan(skData, target='Name')
