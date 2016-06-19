#-*- coding=utf8 -*-
import math


"""
--------------------------
各層の計算をするための関数
--------------------------
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

"""
--------------------------------------
ここからバックプロバケーション用の関数
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

#各層のδの計算
def Delta_Func(x, w, old_delta):
    new_delta = []
    temp_s = []
    #ロジスティックの微分に通す
    x = Dif_Logistic_Func(x)
    #(W,δ)
    for i in range(len(w)):
        temp_t = 0
        for j in range(len(w[0])):
            temp_t += w[i][j] * old_delta[i]
            temp_s.append(temp_t)
    #Δ=f'(x)・temp_s
    for i in range(len(x)):
        new_delta.append(x[i] * temp_s[i])
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

#layerクラス
class Layer:
    """
    ノードを保有する層を作るクラス
    """
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

      
"""
---------------------------
    ここからメイン処理
---------------------------
"""

if __name__ == '__main__':
    print "\n----- Start program -----\n"
    print "---------------------"

    #重み行列の作成
    MakeWeight = lambda x,y:[[1.0 for i in range(x)] for j in range(y)]
    #すべて重み1の行列(２次元配列)を様々な大きさで用意
    w1 = MakeWeight(4,3)
    w2 = MakeWeight(4,3)
    w3 = MakeWeight(4,3)
    w4 = MakeWeight(4,3)
    for z in range(100):
        #１層目(入力層)
        Input = Layer()
        Input.node = [0.3, 0.7, 0.1]
        print "Layer1's node =",
        print Input.node
        print "---------------------"
    
        #２層目(中間層)
        Layer2 = Layer()
        Input.node.append(1) #バイアス
        Layer2.node = FullConect_Func(Input.node, w2)
        print "Layer2's node before Do_Logistic = ",
        print Layer2.node
        Layer2.Do_Logistic()
        print "Layer2's node =",
        print Layer2.node
        print "---------------------"
    
        #３層目(中間層)
        Layer3 = Layer()
        Layer2.node.append(1) #バイアス
        Layer3.node = FullConect_Func(Layer2.node, w3)
        print "Layer3's node before Do_Logistic = ",
        print Layer3.node
        Layer3.Do_Logistic()
        print "Layer3's node =",
        print Layer3.node
        print "---------------------"
        
        #４層目(出力層)
        Layer4 = Layer()
        Layer3.node.append(1) #バイアス
        Layer4.node = FullConect_Func(Layer3.node, w4)
        print "Layer4's node before Do_Softmax = ",
        print Layer4.node
        Layer4.Do_Softmax()
        print "After Softmax =",
        print Layer4.node
        print "---------------------"
    
        print "\n----------- Start BP -----------\n"
        #誤差を出す
        d = [1.0, 0.0, 0.0]
        delta_4 = First_Delta_Func(Layer4.node, d)
        print "delta4 =",
        print delta_4
        w4 = FC_Update_Func(Layer3.node, delta_4, w4)
        print "new w4 =",
        print w4
        print "---------------------"
        #更新後のw4を使用
        delta_3 = Delta_Func(Layer3.node, w4, delta_4)
        print "delta3 =",
        print delta_3
        w3 = FC_Update_Func(Layer2.node, delta_3, w3)
        print "new w3=",
        print w3
        print "---------------------"
        #更新後のw3を使用
        delta_2 = Delta_Func(Layer2.node, w3, delta_3)
        print "delta2 =",
        print delta_2
        w3 = FC_Update_Func(Input.node, delta_2, w2)
        print "new w2=",
        print w2
        print "---------------------"
    
