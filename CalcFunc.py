#-*- coding:utf-8 -*-
import math
import sys
"""
------------------------------
　各層の計算をするための関数
------------------------------
"""

#ロジスティック関数(活性化関数)
def Logistic_Func(x):
    return 1.0 / float((1.0 + math.exp(-x)))

#全結合
def FullConect_Func(x, w):
    #行列の掛け算ができないときのエラー処理
    if len(x) > len(w[0]):
        print "Multiple Faild in All bind layer."
        return 0.0
    next_node = []
    #ノードと重みの内積を求める
    for i in w:
        temp = 0.0
        for j in range(len(x) - 1):
            temp += x[j] * i[j]
        next_node.append(temp)
    return next_node

#畳み込み層
def Conv_Func(x, w):
    #行列の掛け算ができないときのエラー処理
    if len(x) > len(w[0]):
        print "Multiple Faild in Convolution layer."
        return 0.0
    next_node = []
    counter = 0
    #畳み込み処理
    for i in w:
        temp = 0.0
        #バイアス項抜きで内積を取る
        for j in range(counter, counter+len(i)-1):
            temp += x[j] * i[j-counter]
        #バイアス
        temp += i[len(i)-1]
        next_node.append(temp)
        counter += 1
    return next_node

#プーリング層
def Pooling_Func(x, kernel_size):
    next_node = []
    counter = 0
    #カウンターが配列の最後に行くまで
    while counter < len(x):
        #最大値を格納しておく変数
        max_temp = 0
        #カーネル内の最大値を次のノード配列に追加
        for i in range(counter, counter + kernel_size):
            if max_temp < x[i]:
                max_temp = x[i]
        next_node.append(max_temp)
        counter += kernel_size
    return next_node

#重み行列の作成
def MakeWeight(x, y):
    return [[1.0 for i in range(x)] for j in range(y)]

#データの読み込み
def Open_data(filename):
    input_file = open(filename)
    temp = input_file.read().split('\n')
    input_node = []
    for i in temp:
        if i != '':
            input_node.append(map(int, i.split(',')))
    return input_node
