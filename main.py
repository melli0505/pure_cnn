import numpy as np
from layer.convolution import Convolution
from layer.neural_network import FullyConnected


# 1. get training data
with open("data/mnist_train.csv") as file:
    csv_data = []
    for line in file.readlines():
        csv_data.append(line.split(","))

csv_data = csv_data[1:]


mnist_np = np.array(csv_data, dtype=np.int32)


train_y = mnist_np[:, 0]
train_x = mnist_np[:, 1:].reshape(59999, 1, 28, 28)

hiddens = [
    {"name": "dense1", "units": 128},
    {"name": "dense2", "units": 64},
    {"name": "dense3", "units": 32},
    {"name": "dense4", "units": 16},
    {"name": "dense4", "units": 10},
]

NN = FullyConnected(input_layer=np.ones((18432, )), hidden_layer=hiddens)

epochs = 10000

for i in range(10):

    # 2. convolution
    conv2d1 = Convolution(train_x[i], filter_num=8, kernel_size=(3, 3))
    conv1_output = conv2d1.convolution(train_x[i], strides=1, padding=False)
    print("conv1: ", conv1_output.shape)
    conv2d2 = Convolution(conv1_output, filter_num=4, kernel_size=(3, 3))
    conv2_output = conv2d2.convolution(conv1_output, strides=1, padding=False)
    print("conv2: ", conv2_output.shape)

    # 3. flatten

    flatten = conv2_output.reshape(
        conv2_output.shape[0] * conv2_output.shape[1] * conv2_output.shape[2]
    )

    print("flatten: ", flatten.shape)

    # 4. nn
    ff_out = flatten
    for j in range(epochs + 1):
        ff_out = NN.feedforward(flatten)
        cost = NN.calc_cost(train_y[i])
        NN.calc_derivatives(train_y[i])
        NN.backpropagation(0.5)
        if j % 1000 == 0:
            print("epoch" + str(j) + ": ", cost)

    print(f"+ image [{i}]   |   prediction : ", np.argmax(ff_out), "\t label: ", train_y[i] )
