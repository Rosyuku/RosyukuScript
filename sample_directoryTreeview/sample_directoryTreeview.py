#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 23:52:00 2018

@author: kazuyuki
"""

import RosyukuScript.RosyukuScript as rs

if __name__ == "__main__":
   
   path = "/home/kazuyuki/anaconda3/lib/python3.6/site-packages/RosyukuScript"

   df = rs.directoryTreeview(path)
   
   df.to_csv("fileinfo.csv")