import numpy as np
from Network import NeuralNetwork

from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

seed = 1337

# Activation function(s) and it's derivatives
sigmoid = lambda x: 1 / (1 + np.exp(- x))
grad_sigmoid = lambda x: np.multiply(np.exp(- x), np.power((1 + np.exp(- x)), -2))

# Cost functions and it's derivatives
# Mean squared error
mean_squared_error = lambda target, pred: (pred - target)**2 / target.shape[0]
grad_MSE = lambda target, pred: 2 * (pred - target) / target.shape[0]
# Cross-entropy (i.e. log-likelihood)
cross_entroy = lambda target, pred: - (target - pred) / target.shape[0]
grad_cross_ent = lambda target, pred: - (target - pred) / target.shape[0]


"""
Test if we can reproduce the XOR, OR and AND binary gate.
"""
inputs = np.array([
    [0, 0, 1, 1],
    [0, 1, 0, 1]
])
target = np.array([ 
    [0, 1, 1, 0],   # xor
    [0, 1, 1, 1],   # or
    [0, 0, 0, 1]    # and
])

neural_network = NeuralNetwork(2, random_state=seed)
neural_network.add_layer(16, (sigmoid, grad_sigmoid))
neural_network.add_layer(3, (sigmoid, grad_sigmoid))

epochs = 10000
size_batches = 2
learning_rate = lambda x: 0.1
neural_network.train(inputs, target, grad_cross_ent, epochs, learning_rate, size_batches)

output = neural_network.predict(inputs)
print(output)
print(output.round())
