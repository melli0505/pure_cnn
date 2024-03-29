import numpy as np
from .softmax import Softmax


class FullyConnected:
    def __init__(self, input_layer, hidden_layer: list, output_layer=0) -> None:
        self.hidden_layer = hidden_layer
        self.input_layer = input_layer
        self.output_layer = output_layer

        self.L = len(hidden_layer)

        self.parameters = {}

        self.parameters["w" + str(1)] = (
            np.random.randn(input_layer.shape[0], hidden_layer[0]["units"]) * 0.01
        )
        self.parameters["b" + str(1)] = np.ones((hidden_layer[0]["units"],))
        self.parameters["out" + str(1)] = np.ones((hidden_layer[0]["units"], 1))
        self.parameters["net" + str(1)] = np.ones((hidden_layer[0]["units"], 1))

        for i in range(1, len(hidden_layer)):
            self.parameters["w" + str(i + 1)] = (
                np.random.randn(hidden_layer[i - 1]["units"], hidden_layer[i]["units"])
                * 0.01
            )
            self.parameters["b" + str(i + 1)] = np.ones((hidden_layer[i]["units"],))
            self.parameters["out" + str(i + 1)] = np.ones((hidden_layer[i]["units"], 1))
            self.parameters["net" + str(i + 1)] = np.ones((hidden_layer[i]["units"], 1))

        self.parameters["c"] = 1
        self.parameters["s"] = []
        self.derivatives = {}

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-1 * x))

    def sigmoid_prime(self, x):
        return self.sigmoid(x) * (1.0 - self.sigmoid(x))

    def softmax(self, x):
        exp_a = np.exp(x)
        return exp_a / np.sum(exp_a, axis = 0)
    
    def softmax_prime(self, x, y):
        gradient = np.zeros(10)
        gradient[y] = -1 / self.parameters["s"][y]

        for i, grad in enumerate(gradient):
            exp_t = np.exp(self.parameters["out" + str(self.L)])
            s = np.sum(exp_t)


    
    
    def feedforward(self, input_layer):
        """
        w = weight matrix [   ]
        x = value matrix [   ]
        b = bias
        """
        self.parameters["out0"] = input_layer

        for l in range(1, self.L + 1):
            self.parameters["net" + str(l)] = np.add(
                np.dot(
                    self.parameters["out" + str(l - 1)],
                    self.parameters["w" + str(l)],
                ),
                self.parameters["b" + str(l)],
            )

            self.parameters["net" + str(l)] = (
                np.dot(
                    self.parameters["out" + str(l - 1)], self.parameters["w" + str(l)]
                )
                + self.parameters["b" + str(l)]
            )

            self.parameters["out" + str(l)] = self.sigmoid(
                self.parameters["net" + str(l)]
            )

        self.parameters["s"] = self.softmax(self.parameters["out" + str(self.L)])
        return self.parameters["s"]

    def calc_derivatives(self, y):
        # 1. output layer 쪽 미분계수 구하기
        y_mat = np.zeros((10, ))
        y_mat[y] = 1
        # dzL
        self.derivatives["ds"] = y_mat - self.parameters["s"]
        self.derivatives["dz" + str(self.L)] = self.parameters["out" + str(self.L)] - (self.derivatives["ds"] * (1 - self.derivatives["ds"]))
        # dWL
        self.derivatives["dW" + str(self.L)] = (
            self.derivatives["dz" + str(self.L)]
            * self.parameters["out" + str(self.L)]
            * (1 - self.parameters["out" + str(self.L)])
            * np.transpose([self.parameters["out" + str(self.L - 1)]])
        )
        # dbL
        self.derivatives["db" + str(self.L)] = self.derivatives["dz" + str(self.L)]

        # 2. hidden layer 쪽 미분계수 구하기
        for l in range(self.L - 1, 0, -1):

            self.derivatives["dz" + str(l)] = np.dot(
                self.derivatives["dz" + str(l + 1)],
                np.transpose(self.parameters["w" + str(l + 1)]),
            ) * self.sigmoid_prime(self.parameters["net" + str(l)])

            self.derivatives["dW" + str(l)] = np.dot(
                self.derivatives["dz" + str(l)],
                np.transpose(self.parameters["out" + str(l)]),
            )
            self.derivatives["db" + str(l)] = self.derivatives["dz" + str(l)]



    def backpropagation(self, lr):
        for l in range(1, self.L + 1):
            self.parameters["w" + str(l)] -= lr * self.derivatives["dW" + str(l)]

    def calc_cost(self, y):
        # mean square error
        self.parameters["c"] = -np.log(self.parameters["s"][y])
        #-(y*np.log(self.parameters['out' + str(self.L)]) + (1-y)*np.log( 1 - self.parameters['out' + str(self.L)]))
        
        # (1 / 2) * np.sum(
        #     np.subtract(y, self.parameters["out" + str(self.L)]) ** 2
        # )

        # print(self.parameters["out" + str(self.L)].shape)
        return self.parameters["c"]


# hidden_layer = [{"name": "1", "units": 2}, {"name": "1", "units": 2}]
# input_layer = np.array([0.05, 0.1])
# output_layer = []
# fc = FullyConnected(input_layer, hidden_layer, output_layer)
# print(fc.parameters)

# fc.feedforward(input_layer)
# print(fc.parameters)

# w1 = fc.parameters['w1']
# b1 = fc.parameters['b1']
# net1 = fc.parameters['net1']
# out1 = fc.parameters['out1']

# w2 = fc.parameters['w2']
# b2 = fc.parameters['b2']

"""
# 1. feed forward
fc = FullyConnected()
x = np.array([0.05, 0.1])
w = np.array([[0.15, 0.2], [0.25, 0.3]])
b = np.array([0.35, 0.35])
hidden1 = fc.feedforward(x, w, b)
print("hidden1 output: ", hidden1)

w2 = np.array([[0.4, 0.45], [0.5, 0.55]])
b2 = np.array([0.6, 0.6])

output = fc.feedforward(hidden1, w2, b2)
print("output: ", output)


# 2. calculate cost
y = np.array([0.01, 0.99])
cost = fc.calcCost("mse", output, y)
print(cost)


# 3. back propagation


# 1. feed forward
fc = FullyConnected()
x = np.array([4, 3, 5])
w = np.array([[0.2, 0.1, 0.4], [0.5, 0.3, 0.1]])
b = np.array([0.2, 0.1])
hidden1 = fc.feedforward(x, w, b)
print(hidden1)

w2 = np.array([[0.2, 0.3], [0.4, 0.1]])
b2 = np.array([0.1, 0.2])

hidden2 = fc.feedforward(hidden1, w2, b2)
print(hidden2)

w3 = np.array([[0.1, 0.2]])
b3 = np.array([0.1])
output = fc.feedforward(hidden2, w3, b3)
print(output)


# 2. calculate cost
y = np.array([1])
cost = fc.calcCost("mse", output, y)
print(cost)


# 3. back propagation

"""
