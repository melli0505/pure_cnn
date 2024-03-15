import numpy as np
from layer.convolution import Convolution


# 1. get training data
with open("data/mnist_train.csv") as file:
    csv_data = []
    for line in file.readlines():
        csv_data.append(line.split(","))

csv_data = csv_data[1:]


mnist_np = np.array(csv_data, dtype=np.int32)


train_y = mnist_np[:, 0]
train_x = mnist_np[:, 1:].reshape(60000, 1, 28, 28)


for image in train_x[:1]:

    # 2. convolution
    conv2d1 = Convolution(image, filter_num=8, kernel_size=(3, 3))
    conv1_output = conv2d1.convolution(image, strides=1, padding=False)
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


hiddens = [
    {"name": "dense1", "units": 128},
    {"name": "dense2", "units": 64},
    {"name": "dense3", "units": 32},
    {"name": "dense4", "units": 16},
]
