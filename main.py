#-*- coding=utf8 -*-
from libDL import *
      
if __name__ == '__main__':

    print "----------------------"
    print "    Start program   "
    print "----------------------"
    
    #ファイルからの入力値の読み取り
    file_name = sys.argv[1]
    train_name = sys.argv[2]
    input_node = Open_data(file_name)
    #教師データ
    d = Open_data(train_name)

    #すべて重み1の行列(２次元配列)を用意
    w = []
    #層の数だけ重みを作成する
    for i in range(2):
        w.append(MakeWeight(len(input_node[0])+1, len(input_node[0])))
    w.append(MakeWeight(len(input_node[0])+1, 3))

    Input = Layer()
    Layer2 = Layer()
    Layer3 = Layer()
    Layer4 = Layer()
    #inputの数だけ学習させる
    for z in range(len(input_node)):
        #１層目(入力層)
        Input.node = input_node[z]
        #２層目(中間層)
        Pass_FC(Input, Layer2, w[0])
        #３層目(中間層)
        Pass_FC(Layer2, Layer3, w[1])
        #４層目(出力層)
        Pass_FC_Out(Layer3, Layer4, w[2])
        print "Num : %d Output = [" % z, 
        for i in Layer4.node:
            print " %.2f " % i,
        print ']'

        """
        ----------
        ここからBP
        ----------
        """
        #誤差を出す
        delta_4 = First_Delta_Func(Layer4.node, d[z])
        w[2] = FC_Update_Func(Layer3.node, delta_4, w[2])
        #更新後のw4を使用
        delta_3 = Delta_Func(Layer3.node, w[2], delta_4)
        w[1] = FC_Update_Func(Layer2.node, delta_3, w[1])
        #更新後のw3を使用
        delta_2 = Delta_Func(Layer2.node, w[1], delta_3)
        w[0] = FC_Update_Func(Input.node, delta_2, w[0])
    
