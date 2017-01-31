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
        if (expdata[column].dtypes == int) or (expdata[column].dtypes == float) or (expdata[column].dtypes == 'int64'):
            pass
        
        #カテゴリデータはバイナリ化
        elif expdata[column].dtypes == object:
            if len(expdata[column].unique()) > 2:
                temp = pd.DataFrame(index=expdata[column].index, columns=column + " = "  + expdata[column].unique()
                , data=sp.label_binarize(expdata[column], expdata[column].unique()))
                expdata = pd.concat([expdata, temp], axis=1)
            else:
                temp = pd.DataFrame(index=expdata[column].index, columns=[column + " = "  + expdata[column].unique()[0]]
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
    
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ete3 import Tree, TreeStyle, TextFace, PieChartFace, ImgFace

def eteview(bunch, clf, ymax=30, figext="jpg", outfilename="mytree.png", outfiledpi=300, fontsize=15):
    """
    ＜概要＞
    scikit-learnの決定木をeteを使って可視化するファンクション
     
    ＜引数＞
    bunch：scikit-learn形式に変換したデータ
    clf:決定木オブジェクト（学習済み）
    ymax：ヒストグラムのy軸の最大値(デフォルト30)
    figext：クラス画像ファイルの拡張子（デフォルトjpg)
    outfilename：出力画像ファイル名（デフォルトmytree.png）
    outfiledpi：出力画像ファイルの解像度（デフォルト300）
    fontsize：グラフの文字フォント（デフォルト15）
     
    ＜出力＞
    None：出力なし
    """
    
    #データフレーム作成
    df = pd.DataFrame(data=bunch.data, columns=bunch.feature_names)
    
    #各カラムの最大値と最小値を取得（グラフ作成時に必要になる）
    maxList = df.max()
    minList = df.min()
    
    #各データの到達ノードIDを取得
    df['#NAMES'] = clf.tree_.apply(bunch.data.astype(np.float32))
    leafList = df['#NAMES'].unique()
    leafList.sort()    

    #グラフで使う色（好みで変えてください）
    cmap = plt.get_cmap('gist_rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(bunch.feature_names))]
    deffont = plt.rcParams["font.size"]
    #グラフのフォントサイズ
    plt.rcParams["font.size"] = fontsize
    
    #到達ノードIDごとに要素ごとのヒストグラムを作成し保存
    for i in leafList:
        tdf = df[df['#NAMES'] == i]   
        #1つのグラフには16要素まで表示
        for k in range(bunch.feature_names.shape[0] // 16 + 1):
            fig = plt.figure(figsize=(3*len(bunch.feature_names), 3))
            
            for j, c in enumerate(bunch.feature_names[16*k:16*k+15]):
                ax = fig.add_subplot(1, min(16, len(bunch.feature_names)), j+1)
                tdf[c].plot.hist(title=c, color=colors[j],
                bins=np.arange(minList[c]-0.1, maxList[c]+0.1, (maxList[c]-minList[c])/30), xlim=(minList[c]-0.1, maxList[c]+0.1), ylim=(0, ymax))
                
            fig.tight_layout()
            fig.savefig(str(i) + "_" + str(k) + ".jpg")
            plt.close(fig)
    
    #eteのTreeインスタンスを構築
    tree = Tree()
    
    #treeオブジェクトを構築する
    for i in range(clf.tree_.node_count):
        #ルートノードの名称は0とする
        if i == 0:
            tree.name = str(0)
        
        #親ノードを設定
        node = tree.search_nodes(name=str(i))[0]
        
        #ノードごとに配分の円グラフを作成
        Pie = PieChartFace(percents=clf.tree_.value[i][0] / clf.tree_.value[i].sum() * 100
        , width=clf.tree_.n_node_samples[i]
        , height=clf.tree_.n_node_samples[i])
        Pie.opacity = 0.8
        Pie.hz_align = 1
        Pie.vt_align = 1
        
        #円グラフをセット
        node.add_face(Pie, column=2, position="branch-right")

        #左側の子ノードに関する処理
        if clf.tree_.children_left[i] > -1:
            #ノード名称はtreeのリストIDと一致させる
            node.add_child(name=str(clf.tree_.children_left[i]))
            #子ノードに移る
            node = tree.search_nodes(name=str(clf.tree_.children_left[i]))[0]
            #分岐条件を追加
            node.add_face(TextFace(bunch.feature_names[clf.tree_.feature[i]]), column=0, position="branch-top")
            node.add_face(TextFace(u"≦" + "{0:.2f}".format(clf.tree_.threshold[i])), column=1, position="branch-bottom")
            #親ノードに戻っておく
            node = tree.search_nodes(name=str(i))[0]
        
        #右側の子ノードに関する処理（上記と同様）
        if clf.tree_.children_right[i] > -1:
            node.add_child(name=str(clf.tree_.children_right[i]))
            node = tree.search_nodes(name=str(clf.tree_.children_right[i]))[0]
            node.add_face(TextFace(bunch.feature_names[clf.tree_.feature[i]]), column=0, position="branch-top")
            node.add_face(TextFace(">" + "{0:.2f}".format(clf.tree_.threshold[i])), column=1, position="branch-bottom")
            node = tree.search_nodes(name=str(i))[0]
        
        #リーフノードに関する処理
        if clf.tree_.children_left[i] == -1 and clf.tree_.children_right[i] == -1:
            
            #リーフの情報を取得
            text1 = "{0:.0f}".format(clf.tree_.value[i][0][np.argmax(clf.tree_.value.T, axis=0)[0][i]] / clf.tree_.n_node_samples[i] * 100) + "%"
            text2 = "{0:.0f}".format(clf.tree_.value[i][0][np.argmax(clf.tree_.value.T, axis=0)[0][i]]) +"/"+ "{0:.0f}".format(clf.tree_.n_node_samples[i])
            
            #リーフの情報を書き込み
            node.add_face(TextFace(bunch.target_names[np.argmax(clf.tree_.value.T, axis=0)[0][i]])
            , column=4, position="branch-right")
            node.add_face(TextFace(text1)
            , column=4, position="branch-right")
            node.add_face(TextFace(text2)
            , column=4, position="branch-right")
            
            #クラスに対応した画像を設置
            imgface = ImgFace(bunch.target_names[np.argmax(clf.tree_.value.T, axis=0)[0][i]] + "." + figext, height=80)
            imgface.margin_left = 10
            imgface.margin_right = 10            
            node.add_face(imgface, column=3, position="branch-right")
            
            for k in range(bunch.feature_names.shape[0] // 16 + 1):
                #作成したヒストグラムを設置
                imgface2 = ImgFace(str(i) + "_" + str(k) + ".jpg", height=150)
                node.add_face(imgface2, column=4+k, position="aligned")
    
    #不要な要素を表示しないように設定    
    ts = TreeStyle()
    ts.show_leaf_name = False
    ts.show_scale = False
    
    #ファイル保存
    tree.render(outfilename, dpi=outfiledpi, tree_style=ts)
    
    #グラフのフォントサイズを元に戻す
    plt.rcParams["font.size"] = deffont