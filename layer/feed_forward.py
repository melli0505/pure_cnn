import numpy as np


class FullyConnected:
    def __init__(self) -> None:
        pass

    def feedforward(self, input_layer, hidden_units, bias):
        """
        w = weight matrix [   ]
        x = value matrix [   ]
        b = bias
        """
        hidden_output = []

        for u in range(len(hidden_units)):
            unit_output = 0
            for i in range(len(input_layer)):
                unit_output += input_layer[i] * hidden_units[u][i]
            unit_output += bias[u]
            hidden_output.append(unit_output)

        return np.array(hidden_output)

    def backpropagation(self):
        pass

    def calcCost(self, func: str, y_hat, y):
        if func == "mse":
            return (y - y_hat) ** 2 / len(y)
        elif func == "binaryCrossEntropy":
            pass
        elif func == "crossEntropy":
            pass

    def optimization(self):
        pass


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

y = np.array([1])
cost = fc.calcCost("mse", output, y)
print(cost)
