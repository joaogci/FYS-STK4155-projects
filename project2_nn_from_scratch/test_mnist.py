import sys
sys.path.append('../')

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
Test if we can determine handwritten numbers from MNist dataset.
"""
# Extract data for ScikitLearn
digits = load_digits()
data = digits.data
target = digits.target

X_train, X_test, y_train_wrong_dim, y_test = train_test_split(data, target, test_size=0.25, random_state=seed)

y_train = np.zeros((y_train_wrong_dim.shape[0],10))
for i in range(len(y_train_wrong_dim)):
    y_train[ i, y_train_wrong_dim[i] ] = 1

y_train = y_train.T

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train).T
X_test = scaler.transform(X_test).T

neural_network = NeuralNetwork(64, random_state=seed)
neural_network.add_layer(16, sigmoid())
neural_network.add_layer(16, sigmoid())
neural_network.add_layer(10, sigmoid())

epochs = 1000
size_batches = 5

start = 4
end = 1
learning_rate = lambda x: start - (start - end) * x / epochs

learning_rate = lambda x: 3
neural_network.train(X_train, y_train, grad_cross_entropy, epochs, learning_rate, size_batches)

pred = neural_network.predict(X_test)
pred = np.argmax(pred, axis=0)

print(f"accuracy: {np.sum(pred == y_test)}/{y_test.shape[0]} = {np.sum(pred == y_test)/y_test.shape[0]:.3f}")
