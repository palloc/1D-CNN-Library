#-*- coding=utf8 -*-
from CalcFunc import *

#FC層の計算関数
def Pass_FC(old_layer, new_layer, w):
    new_layer.bp_node = old_layer.node
    old_layer.node.append(1)
    new_layer.node = FullConect_Func(old_layer.node, w)
    new_layer.Do_Logistic()

#出力層のFC関数
def Pass_FC_Out(old_layer, new_layer, w):
    old_layer.node.append(1)
    new_layer.node = FullConect_Func(old_layer.node, w)
    new_layer.Do_Softmax()

"""
--------------------------------------
    バックプロバケーション用の関数
--------------------------------------
"""

#ロジスティック関数の微分
def Dif_Logistic_Func(x):
    new_node = []
    for i in x:
        new_node.append(Logistic_Func(i) * (1 - Logistic_Func(i)))
    return new_node

#最初のクロスエントロピーの微分δの計算
def First_Delta_Func(x, d):
    result = []
    for i in range(len(x)):
        result.append( x[i] - d[i] )
    return result

#全結合層のδの計算
def Delta_Func(node, w, delta):
    new_delta = []
    temp_s = []
    #ロジスティックの微分に通す
    node = Dif_Logistic_Func(x)
    #(W,δ)
    for i in range(len(w)):
        temp_t = 0
        for j in range(len(w[0])):
            temp_t += w[i][j] * delta[i]
            temp_s.append(temp_t)
    #Δ=f'node)・temp_s
    for i in range(len(node)):
        new_delta.append(node[i] * temp_s[i])
    return new_delta

#畳み込み層のδの計算
def Conv_Delta(layer, w, delta):
    new_delta = []
    temp_s = 0.0
    #ロジスティックの微分に通す
    node = Dif_Logistic_Func(x)

    for i in range(len(layer.old_node)):
        for j in range(len(w)):
            if (i-j >= 0) and (i-j <= len(delta)):
                temp_s += delta[i-j] * w[j]
        new_delta.append(temp_s)
        temp_s = 0.0
    return new_delta

#プーリング層のδ
def Max_Pool_Delta(layer, delta):
    new_delta = []
    #そのまま通す。落としたノードは0とする。
    counter = 0
    while counter > len(layer.node):
        for i in range(counter, counter+layer.kernel_size):
            if layer.old_node[i] == layer.node[counter]:
                new_delta.append(delta[counter])
            else:
                new_delta.append(0)
    return new_delta

#出力層の重みの更新
def FC_Update_Func(x, delta, w):
    new_w = []
    for i in range(len(w)):
        temp = []
        for j in range(len(w[i])):
            temp.append(w[i][j] - delta[i] * x[j])
        new_w.append(temp)
    return new_w

#畳み込み層の重みの更新
def Conv_Update_Func(node, delta, w):
    new_w = []
    temp = 0.0
    for i in range(len(w)):
        for j in range(len(node)-len(w)+1):
            temp += delta[j] * node[j+i]
        new_w.append(temp)
        temp = 0.0
    return new_w

#layerクラス
class Layer:
    """
    ノードを保有する層を作成するクラス
    """
    #コンストラクタ
    def __init__(self):
        #ノード情報を保持する配列
        self.node = []
        #プーリング層のbpをする際に用いる一つ前のノードの情報とカーネルサイズ
        self.kernel_size = 0
        self.bp_node = []
    #全ノードをロジスティック関数に通す
    def Do_Logistic(self):
        self.node = map(Logistic_Func, self.node)
    #ソフトマックス関数(出力層での活性化関数)
    def Softmax_Func(self, x):
        return math.exp(x) / sum(map(math.exp, self.node))
    #ソフトマックス関数を全ノードに通す
    def Do_Softmax(self):
        Sum = sum(map(math.exp, self.node))
        temp = lambda x:math.exp(x) / Sum
        self.node = map(temp, self.node)

