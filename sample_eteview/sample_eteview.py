# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 23:26:12 2017

@author: Wakasugi Kazuyuki
"""

import RosyukuScript.RosyukuScript as rs
from sklearn.datasets import load_iris
from sklearn import tree
import importlib
importlib.reload(rs)

if __name__ == "__main__":
    
    #irisデータの読み込み
    iris = load_iris()
     
    #決定木学習
    clf = tree.DecisionTreeClassifier(max_depth=3)
    clf.fit(iris.data, iris.target)
    
    #eteview実行
    rs.eteview(iris, clf)