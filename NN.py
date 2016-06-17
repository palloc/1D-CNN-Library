#-*- coding=utf8 -*-
import math

#ロジスティック関数(活性化関数)
def Logistic_Func(x):
    return 1.0 / float((1.0 + math.exp(-x)))


# layerクラス
class Layer:
    """ノードを保有する層を作るクラス"""
    #何層目かを表すid
    layer_id = 1
    #コンストラクタ
    def __init__(self):
        #ノード情報を保持する配列
        self.node = []
        Layer.layer_id += 1
    #全ノードをロジスティック関数に通す
    def Do_Logistic(self):
        self.node = map(Logistic_Func, self.node)
    #ソフトマックス関数(出力層での活性化関数)
    def Softmax_Func(self, x):
        return math.exp(x) / sum(map(math.exp, self.node))
    #ソフトマックス関数を全ノードに通す
    def Do_Softmax(self):
        Sum = sum(map(math.exp, self.node))
        temp = lambda x:math.exp(x)/Sum
        self.node = map(temp, self.node)

#全結合
def AllBind_Func(x, w):
    #行列の掛け算ができないときのエラー処理
    if len(x) != len(w[0]):
        print "Multiple Faild in All bind layer."
        return 0.0
    next_node = []
    for i in w:
        temp = 0.0
        for j in range(len(x)):
            temp += x[j] * i[j]
        next_node.append(temp)
    return next_node

    
#メイン
if __name__ == '__main__':
    print "\n----- Start program -----\n"
    print "---------------------"
    w1 = [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]
    w2 = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]

    #１層目(入力層)
    Input = Layer()
    Input.node = [0.3, 0.7, 0.1, 0.3]
    print "Layer1's node = ",
    print Input.node
    print "---------------------"

    #２層目(中間層)
    Layer2 = Layer()
    Layer2.node = AllBind_Func(Input.node, w1)
    print "Layer2's node before Do_Logistic = ",
    print Layer2.node
    Layer2.Do_Logistic()
    print "Layer2's node = ",
    print Layer2.node
    print "---------------------"

    #３層目(出力層)
    Layer3 = Layer()
    Layer3.node = AllBind_Func(Layer2.node, w2)
    print "Layer3's node before Do_Logistic = ",
    print Layer3.node
    Layer3.Do_Softmax()
    print "After Softmax = ",
    print Layer3.node
    print "---------------------"
    
