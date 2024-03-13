import numpy as np


class FullyConnected:
    def __init__(self) -> None:
        self.weights = []
        pass

    def feedforward(self, input_layer, hidden_units, bias):
        """
        w = weight matrix [   ]
        x = value matrix [   ]
        b = bias

        logistic function = 1 / (1 + e**(-x))
        """
        hidden_output = []

        for u in range(len(hidden_units)):
            unit_output = 0
            for i in range(len(input_layer)):
                unit_output += input_layer[i] * hidden_units[u][i]
            unit_output += bias[u]

            hidden_output.append(unit_output)

        output = 1 / (1 + np.exp(np.array(hidden_output) * -1))
        return output

    def backpropagation(self):
        pass

    def calcCost(self, func: str, y_hat, y):
        if func == "mse":
            return np.sum((y - y_hat) ** 2 / len(y))
        elif func == "binaryCrossEntropy":
            pass
        elif func == "crossEntropy":
            pass

    def optimization(self):
        pass


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
