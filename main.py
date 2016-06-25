#-*- coding=utf8 -*-
from function import *
      
"""
---------------------------
        メイン処理
---------------------------
"""

if __name__ == '__main__':
    print "\n----- Start program -----\n"

    print "---------------------"

    #重み行列の作成
    MakeWeight = lambda x,y:[[1.0 for i in range(x)] for j in range(y)]
    
    #ファイルからの入力値の読み取り
    input_file = open("data")
    temp = input_file.read().split('\n')
    input_node = []
    for i in temp:
        if i != '':
            input_node.append(map(int, i.split(',')))

    #すべて重み1の行列(２次元配列)を様々な大きさで用意
    w1 = MakeWeight(len(input_node[0])+1, 3)
    w2 = MakeWeight(len(input_node[0])+1, 3)
    w3 = MakeWeight(len(input_node[0])+1, 3)
    w4 = MakeWeight(len(input_node[0])+1, 3)

    #inputの数だけ学習させる
    for z in range(len(input_node)):
        print "\n----------- Start learning  -----------\n"  
        #１層目(入力層)
        Input = Layer()
        Input.node = input_node[z]
        print "Layer1's node =",
        print Input.node
        print "---------------------"
    
        #２層目(中間層)
        Layer2 = Layer()
        Input.node.append(1) #バイアス
        Layer2.node = FullConect_Func(Input.node, w2)
        #ロジスティック関数に通す
        Layer2.Do_Logistic()
        print "Layer2's node =",
        print Layer2.node
        print "---------------------"
    
        #３層目(中間層)
        Layer3 = Layer()
        Layer2.node.append(1) #バイアス
        Layer3.node = FullConect_Func(Layer2.node, w3)
        #ロジスティック関数に通す
        Layer3.Do_Logistic()
        print "Layer3's node =",
        print Layer3.node
        print "---------------------"
        
        #４層目(出力層)
        Layer4 = Layer()
        Layer3.node.append(1) #バイアス
        Layer4.node = FullConect_Func(Layer3.node, w4)
        #ソフトマックス関数に通す
        Layer4.Do_Softmax()
        print "After Softmax =",
        print Layer4.node
        print "---------------------"
    
        print "\n----------- Start BP -----------\n"
        #誤差を出す
        d = t_data[z]
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
    
