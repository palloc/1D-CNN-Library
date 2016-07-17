#-*- coding=utf8 -*-
from libDL import *
      
if __name__ == '__main__':
    #ファイルからの入力値の読み取り
    file_name = sys.argv[1]
    train_name = sys.argv[2]
    input_node = Open_data(file_name)
    #教師データ
    d = Open_data(train_name)

    #すべて重み1の行列(２次元配列)を用意
    w = []
    #層の数だけ重みを作成する

    conv_kernel = 4
    pool_kernel = 2
    #Conv
    w.append(MakeWeight(4,1))
    #M_Pool
    next_nodelen=(len(input_node[0])-conv_kernel+1)/pool_kernel
    #FC
    w.append(MakeWeight(next_nodelen, 3))
    

    Input = Layer()
    Conv_Layer1 = Layer()
    Pool_Layer1 = Layer()
    Out_Layer1 = Layer()
    count = 0
    #inputの数だけ学習させる
    for z in range(len(input_node)):
        #１層目(入力層)
        Input.node = input_node[z]
        #２層目(中間層)
        Pass_Conv(Input, Conv_Layer1, w[0][0])
        #３層目(中間層)
        Pool_Layer1.bp_node = Conv_Layer1.node
        Pass_Max_Pool(Conv_Layer1, Pool_Layer1, pool_kernel)
        #４層目(出力層)
        Pass_FC_Out(Pool_Layer1, Out_Layer1, w[1])
        
        #結果の出力
        print "Num : %d Output = [" % z, 
        for i in Out_Layer1.node:
            print " %.2f " % i,
        print ']'

        max = [0,0]
        for i in range(len(Out_Layer1.node)):
            if Out_Layer1.node[i] > max[0]:
                max[0] = Out_Layer1.node[i]
                max[1] = i
        if d[z][max[1]] == 1:
            print "correct answer!"
            count += 1
        """
        ----------
        ここからBP
        ----------
        """
        #誤差を出す
        delta_2 = First_Delta_Func(Out_Layer1.node, d[z])
        w[1] = FC_Update_Func(Pool_Layer1.node, delta_2, w[1])
        #更新後のw4を使用
        print Pool_Layer1.node
        delta_1 = Max_Pool_Delta(Pool_Layer1, delta_2)
        print delta_1
        #更新後のw3を使用
        delta_0 = Conv_Delta(Conv_Layer1, delta_1, w[0][0])
        print delta_0
        w[0][0] = Conv_Update_Func(Input.node, delta_1, w[0][0])

    print "%d correct." % count
