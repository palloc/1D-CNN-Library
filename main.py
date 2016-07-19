#-*- coding=utf8 -*-
from libDL import *
      
if __name__ == '__main__':
    #read data from file
    file_name = sys.argv[1]
    train_name = sys.argv[2]
    input_node = Open_data(file_name)
    #training data
    d = Open_data(train_name)

    #Prepare weight
    w = []
    #decide kernel size
    conv_kernel = 4
    pool_kernel = 2
    #Conv weight
    w.append(MakeWeight(4,1))
    #M_Pool weight
    next_nodelen=(len(input_node[0])-conv_kernel+1)/pool_kernel
    #FC weight
    w.append(MakeWeight(next_nodelen, 3))
    
    #Prepare layer object
    Input = Layer()
    Conv_Layer1 = Layer()
    Conv_Layer1.kernel_size = conv_kernel    
    Pool_Layer1 = Layer()
    Pool_Layer1.kernel_size = pool_kernel
    Out_Layer1 = Layer()


    """
    ------------------------------
            start learning
    ------------------------------
    """
    for z in range(len(input_node)):

        Input.node = input_node[z]
        Pass_Conv(Input, Conv_Layer1, w[0][0])
        #Pooling must remember pre node
        Pool_Layer1.bp_node = Conv_Layer1.node
        Pass_Max_Pool(Conv_Layer1, Pool_Layer1, pool_kernel)
        Pass_FC_Out(Pool_Layer1, Out_Layer1, w[1])
        
        #print output node
        print "Num : %d Output = [" % z, 
        for i in Out_Layer1.node:
            print " %.2f " % i,
        print ']'


        """
        ------------------------------
            start back propagation
        ------------------------------
        """

        #Calc delta
        delta_3 = Cross_Entropy(Out_Layer1.node, d[z])
        delta_2 = FC_Delta(Pool_Layer1.node, delta_3, w[1])
        delta_1 = Max_Pool_Delta(Pool_Layer1, delta_2)
        delta_0 = Conv_Delta(Conv_Layer1, delta_1, w[0][0])
        #Update weight
        w[1] = FC_Update(Pool_Layer1.node, delta_3, w[1])
        w[0][0] = Conv_Update(Input.node, delta_1, w[0][0])


    """
    ------------------------------
      start evaluating accuracy
    ------------------------------
    """
    
    Input2 = Layer()
    Conv_Layer2 = Layer()
    Pool_Layer2 = Layer()
    Out_Layer2 = Layer()
    count = 0

    for z in range(len(input_node)):
        Input2.node = input_node[z]
        Pass_Conv(Input2, Conv_Layer2, w[0][0])
        Pass_Max_Pool(Conv_Layer2, Pool_Layer2, pool_kernel)
        Pass_FC_Out(Pool_Layer2, Out_Layer2, w[1])
        
        """
        --------------------------
        　　　　print result
        --------------------------
        """
        max = [0,0]
        for i in range(len(Out_Layer2.node)):
            if Out_Layer2.node[i] > max[0]:
                max[0] = Out_Layer2.node[i]
                max[1] = i
        if d[z][max[1]] == 1:
            count += 1

    print "%.2f%% correct." % (float(count)/float(len(input_node))*100.0)
