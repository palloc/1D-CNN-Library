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

    Input = Layer()
    Layer2 = Layer()
    Layer3 = Layer()
    Layer4 = Layer()
    #inputの数だけ学習させる
    for z in range(len(input_node)):
        print "----------------------"
        print "      Start FFNN      "
        print "----------------------"

        #１層目(入力層)
        Input.node = input_node[z]
        print "Layer1's node =",
        print Input.node
        #２層目(中間層)
        Pass_FC(Input, Layer2, w[0])
        print "Layer2's node = ",
        print Layer2.node
        #３層目(中間層)
        Pass_FC(Layer2, Layer3, w[1])
        print "Layer3's node = ",
        print Layer3.node
        #４層目(出力層)
        Pass_FC_Out(Layer3, Layer4, w[2])
        print "Layer4's node = " ,
        print Layer4.node

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
        w[2] = FC_Update_Func(Layer3.node, delta_4, w[2])
        print "new w4 =",
        print w[2]

        #更新後のw4を使用
        delta_3 = Delta_Func(Layer3.node, w[2], delta_4)
        w[1] = FC_Update_Func(Layer2.node, delta_3, w[1])
        print "\nnew w3=",
        print w[1]

        #更新後のw3を使用
        delta_2 = Delta_Func(Layer2.node, w[1], delta_3)
        w[0] = FC_Update_Func(Input.node, delta_2, w[0])
        print "\nnew w2=",
        print w[0]

    
        print "----------------------"
        print "        Fin BP        "
        print "----------------------"
