# One dimensional CNN library

# Usage

- MakeWeight(x, y)
  - Make weight array. (w[x][y])

- Open_data(filename)
  - Open [filename] file.

## FFNN Function

### Function for calculate node.

- Logistic_Func(x)
  - Calc logistic function. It's used as activation function.

- FullConect_Func(x, w)
  - Full connect layer's Calc function.

- Conv_Func(x, w)
  - Convolution layer's Calc function.

- Max_Pool_Func(x, kernel_size)
  - Max pooling layer's Calc function.

### Pass NN function

- Pass_FC(old_layer, new_layer, w)
  - Pass full connect layer automatically.

- Pass_Conv(old_layer, new_layer, w)
  - Pass convolution layer automatically.

- Pass_Max_Pool(old_layer, new_layer, kernel_size)
  - Pass max pooling layer automatically.
 
- Pass_FC_Out(old_layer, new_layer, w)
  - Pass Output(full connect) layer automatically.

## BPNN Function

- Dif_Logistic_Func(x)
  - Differential logistic function.

- Cross_Entropy(x, d)
  - Calculate cross entropy from output node.

### Calculate delta

- FC_Delta(node, delta, w)
  - Calculate full connect layer's delta. (For next bp node)

- Conv_Delta(node, delta, w)
  - Calculate convolution layer's delta. (For next bp node)

- Max_Pool_Delta(layer, delta)
  - Calculate max pooling layer's delta. (For next bp node)

### Update node

- FC_Update(x, delta, w)
  - Update full connect layer's weight.

- Conv_Update(node, delta, w)
  - Update convolution layer's wieght.

# example
$python [main.py](https://github.com/palloc/NeuralNetwork/blob/master/main.py) learn_data train_data


