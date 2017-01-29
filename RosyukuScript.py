# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn import datasets
import sklearn.preprocessing as sp

def pan2sk(df, target, name="Data"):
    """
    ＜概要＞
    pandasのデータフレームをscikit-learnの入力データに変換する関数
    
    ＜引数＞
    df：データフレーム
    target：目的変数のカラム名
    
    ＜出力＞
    Bunch：scikit-learn形式に変換したデータ
    """
    
    #説明変数のデータ列と目的変数のデータ列に分ける    
    expdata = df[df.columns[df.columns!=target]]
    objdata = df[target].copy()
    
    #説明変数の各データについて変換
    for column in expdata.columns:
        #数値データはそのまま
        if (expdata[column].dtypes == int) or (expdata[column].dtypes == float):
            pass
        
        #カテゴリデータはバイナリ化
        elif expdata[column].dtypes == object:
            temp = pd.DataFrame(index=expdata[column].index, columns=column + " = "  + expdata[column].unique()
            , data=sp.label_binarize(expdata[column], expdata[column].unique()))
            expdata = pd.concat([expdata, temp], axis=1)
            del expdata[column]
            
        #それ以外のデータ（時系列等）は除外
        else:
            del expdata[column]
    
    #説明変数のデータとカラム名を分けておく
    data=np.array(expdata)
    feature_names=np.array(expdata.columns)
    
    #目的変数のデータをシリアル化する
    #数値データはそのまま登録
    if (objdata.dtypes == int) or (objdata.dtypes == float):
        targetData = np.array(objdata)
        target_names = []

    #カテゴリデータはシリアル化して登録
    if objdata.dtypes == object:
        
        le = sp.LabelEncoder()
        le.fit(objdata.unique())
        
        targetData = le.transform(objdata)
        target_names = objdata.unique()

    #データセットの名称を用意
    DESCR = name
    
    #オブジェクト作成
    skData = datasets.base.Bunch(DESCR=DESCR, data=data, feature_names=feature_names, target=targetData, target_names=target_names)
    
    return skData