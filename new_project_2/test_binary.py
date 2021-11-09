import numpy as np
from Network import NeuralNetwork

from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from Activation import *
from Cost import *

seed = 1337

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
neural_network.add_layer(16, sigmoid())
neural_network.add_layer(3, sigmoid())

epochs = 10000
size_batches = 2
learning_rate = lambda x: 0.1
neural_network.train(inputs, target, grad_cross_entropy, epochs, learning_rate, size_batches)

output = neural_network.predict(inputs)
print(output)
print(output.round())
