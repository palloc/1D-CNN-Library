#-*- coding=utf8 -*-
from libDL import *
      
if __name__ == '__main__':
    print "----------------------"
    print "    Start program   "
    print "----------------------"
    
    #ファイルからの入力値の読み取り
    file_name = sys.argv[1]
    input_node = open_data(file_name)
    
    #すべて重み1の行列(２次元配列)を用意
    w = []
    for i in range(len(input_node)):
        w.append(MakeWeight(len(input_node[0])+1, 3))

    #inputの数だけ学習させる
    for z in range(len(input_node)):
        print "----------------------"
        print "      Start FFNN      "
        print "----------------------"

        #１層目(入力層)
        Input = Layer()
        Input.node = input_node[z]
        print "Layer1's node =",
        print Input.node
        print "---------------------"
    
        #２層目(中間層)
        Layer2 = Layer()
        Pass_FC(Input, Layer2, w[0])
        #３層目(中間層)
        Layer3 = Layer()
        Pass_FC(Layer2, Layer3, w[1])
        #４層目(出力層)
        Layer4 = Layer()
        Pass_FC_Out(Layer3, Layer4, w[2])

        print "----------------------"
        print "       Fin FFNN       "
        print "----------------------"

        print "----------------------"
        print "       Start BP       "
        print "----------------------"
        
        #誤差を出す
        #d = t_data[z]
        d = [1,0,0]
        delta_4 = First_Delta_Func(Layer4.node, d)
        w4 = FC_Update_Func(Layer3.node, delta_4, w[2])
        print "new w4 =",
        print w[2]
        print "---------------------"
        #更新後のw4を使用
        delta_3 = Delta_Func(Layer3.node, w[2], delta_4)
        w3 = FC_Update_Func(Layer2.node, delta_3, w[1])
        print "new w3=",
        print w[1]
        print "---------------------"
        #更新後のw3を使用
        delta_2 = Delta_Func(Layer2.node, w[1], delta_3)
        w3 = FC_Update_Func(Input.node, delta_2, w[0])
        print "new w2=",
        print w[0]
        print "---------------------"
    
