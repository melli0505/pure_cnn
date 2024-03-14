import numpy as np


class FullyConnected:
    def __init__(self, input_layer, hidden_layer: list, output_layer) -> None:
        self.hidden_layer = hidden_layer
        self.input_layer = input_layer
        self.output_layer = output_layer

        self.L = len(hidden_layer)

        self.parameters = {}

        self.parameters["w" + str(1)] = (
            np.random.randn(hidden_layer[0]["units"], input_layer.shape[0]) * 0.01
        )
        self.parameters["b" + str(1)] = np.ones((hidden_layer[0]["units"], 1))
        self.parameters["out" + str(1)] = np.ones((hidden_layer[0]["units"], 1))
        self.parameters["net" + str(1)] = np.ones((hidden_layer[0]["units"], 1))

        for i in range(1, len(hidden_layer)):
            self.parameters["w" + str(i + 1)] = (
                np.random.randn(hidden_layer[i - 1]["units"], hidden_layer[i]["units"])
                * 0.01
            )
            self.parameters["b" + str(i + 1)] = np.ones((hidden_layer[i]["units"], 1))
            self.parameters["out" + str(i + 1)] = np.ones((hidden_layer[i]["units"], 1))
            self.parameters["net" + str(i + 1)] = np.ones((hidden_layer[i]["units"], 1))

        self.parameters["c"] = 1
        self.derivatives = {}

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-1 * x))

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
                    self.parameters["w" + str(l)], self.parameters["out" + str(l - 1)]
                ),
                self.parameters["b" + str(l)],
            )
            self.parameters["out" + str(l)] = self.sigmoid(
                self.parameters["net" + str(l)]
            )

    def calc_derivatives(self, y):
        # 1. output layer 쪽 미분계수 구하기
        # -(target_o - out_o) * out_o(1 - out_o) * out_h
        self.parameters["d_out" + str(self.L)] = -(
            y - self.parameters["out" + str(self.L)]
        )
        self.parameters["dw" + str(self.L)] = self.parameters["d_out"] * None
        self.parameters["dz" + str(self.L)] = self.parameters["d_out" + str(self.L)]

        # 2. hidden layer 쪽 미분계수 구하기

    def backpropagation(self):
        pass

    def calc_cost(self, y):
        # mean square error
        self.parameters["c"] = (1 / len(y)) * np.sum(
            np.subtract(y, self.parameters["out" + str(self.L)]) ** 2
        )

    def optimization(self):
        pass


hidden_layer = [{"name": "1", "units": 2}, {"name": "1", "units": 2}]
input_layer = np.array([0.05, 0.1])
output_layer = []
fc = FullyConnected(input_layer, hidden_layer, output_layer)
print(fc.parameters)

fc.feedforward(input_layer)
print(fc.parameters)
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
