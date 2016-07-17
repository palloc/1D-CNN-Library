#-*- coding=utf8 -*-
import math
import sys


#make weight
def MakeWeight(x, y):
    return [[1.0 for i in range(x)] for j in range(y)]

def MakeConvWeight(x):
    return [1.0 for i in range(x)]

#read data
def Open_data(filename):
    input_file = open(filename)
    temp = input_file.read().split('\n')
    input_node = []
    for i in temp:
        if i != '':
            input_node.append(map(int, i.split(',')))
    return input_node
epsilon = 0.5


"""
------------------------------
   Calculate Layer's node
------------------------------
"""

#Logistic function(activation function)
def Logistic_Func(x):
    return 1.0 / float((1.0 + math.exp(-x)))

#Full connect layer's Calc_func
def FullConect_Func(x, w):
    #error process(if can't mult matrix)
    if len(x) < len(w[0]):
        print "Multiple Faild in All bind layer."
        return 0.0
    next_node = []
    #dot product
    for i in w:
        temp = 0.0
        for j in range(len(w)):
            temp += x[j] * i[j]
        temp += x[-1]
        next_node.append(temp)
    return next_node

#Convolution layer's Calc_func
def Conv_Func(x, w):
    #error process(if can't mult matrix)
    if len(x) < len(w):
        print "Multiple Faild in Convolution layer."
        return 0.0
    next_node = []
    counter = 0
    #Convolution
    for i in range(len(x)-len(w)):
        temp = 0.0
        #dot product without bias
        for j in range(counter, counter+len(w)-1):
            temp += x[j] * w[j-counter]
        #bias
        temp += w[len(w)-1]
        next_node.append(temp)
        counter += 1
    return next_node

#Max pooling layer's Calc_func
def Max_Pool_Func(x, kernel_size):
    next_node = []
    counter = 0
    while counter < len(x):
        max_temp = 0.0
        #Add max_node in the kernel to next_node
        for i in range(counter, counter + kernel_size - 1):
            if max_temp < x[i]:
                max_temp = x[i]
        next_node.append(max_temp)
        counter += kernel_size
    return next_node


#Pass full connect layer
def Pass_FC(old_layer, new_layer, w):
    new_layer.bp_node = old_layer.node
    old_layer.node.append(1)
    new_layer.node = FullConect_Func(old_layer.node, w)
    new_layer.Do_Logistic()

#Pass convolution layer
def Pass_Conv(old_layer, new_layer, w):
    new_layer.bp_node = old_layer.node
    old_layer.node.append(1)
    new_layer.node = Conv_Func(old_layer.node, w)
    new_layer.Do_Logistic()

#pass max_pooling layer
def Pass_Max_Pool(old_layer, new_layer, kernel_size):
    new_layer.node = Max_Pool_Func(old_layer.node, kernel_size)
    
#pass full connect layer
def Pass_FC_Out(old_layer, new_layer, w):
    old_layer.node.append(1)
    new_layer.node = FullConect_Func(old_layer.node, w)
    new_layer.Do_Softmax()


"""
-----------------------------------
　　Function for back-propagation
-----------------------------------
"""

#Differential logistic function
def Dif_Logistic_Func(x):
    new_node = []
    for i in x:
        new_node.append(Logistic_Func(i) * (1 - Logistic_Func(i)))
    return new_node

#Calculate cross entropy
def Cross_Entropy(x, d):
    result = []
    for i in range(len(x)):
        result.append( x[i] - d[i] )
    return result

#Calculate full connect layer's δ
def FC_Delta(node, delta, w):
    new_delta = []
    temp_s = []
    #Pass differential logistic function
    node = Dif_Logistic_Func(node)
    #(W,δ)
    for i in range(len(w)):
        temp_t = 0
        for j in range(len(w[0])):
            temp_t += w[i][j] * delta[i]
            temp_s.append(temp_t)
    for i in range(len(node)):
        new_delta.append(node[i] * temp_s[i])
    return new_delta

#Calculate convolution layer's δ
def Conv_Delta(layer, w, delta):
    new_delta = []
    temp_s = 0.0
    #Pass differential logistic function    
    node = Dif_Logistic_Func(layer.node)
    for i in range(len(layer.bp_node)):
        for j in range(len(w)):
            if (i-j >= 0) and (i-j < len(delta)):
                temp_s += delta[i-j] * w[j]
        new_delta.append(temp_s)
        temp_s = 0.0
    return new_delta

#Calculate max pooling layer's δ
def Max_Pool_Delta(layer, delta):
    new_delta = []
    #0 replace max node with zero
    counter = 0
    while counter < len(layer.node):
        for i in range(counter, counter+layer.kernel_size):
            if layer.bp_node[i] == layer.node[counter]:
                new_delta.append(delta[counter])
            else:
                new_delta.append(0.0)
        counter += 1
    return new_delta

#Update full connect layer's wight
def FC_Update(x, delta, w):
    new_w = []
    for i in range(len(w)):
        temp = []
        #Update w
        for j in range(len(w[i])):
            temp.append(w[i][j] - epsilon * delta[i] * x[j])
        new_w.append(temp)
    return new_w

#Update convolution layer's wight
def Conv_Update(node, delta, w):
    new_w = []
    temp = 0.0
    for i in range(len(w)):
        #Update w
        for j in range(len(node)-len(w)+1):
            temp += delta[j] * node[j+i]
        new_w.append(w[i] - epsilon * temp)
        temp = 0.0
    return new_w


"""
----------------------
     Layer class
----------------------
"""
class Layer:
    def __init__(self):
        #Array for store the node information
        self.node = []
        #kernel_size(for convolution bp and pooling bp)
        self.kernel_size = 0
        #node information for bp(store pre node)
        self.bp_node = []
    #Do logistic
    def Do_Logistic(self):
        self.node = map(Logistic_Func, self.node)
    #Softmax function(activation function for output layer)
    def Softmax_Func(self, x):
        return math.exp(x) / sum(map(math.exp, self.node))
    #Do softmax
    def Do_Softmax(self):
        Sum = sum(map(math.exp, self.node))
        temp = lambda x:math.exp(x) / Sum
        self.node = map(temp, self.node)

