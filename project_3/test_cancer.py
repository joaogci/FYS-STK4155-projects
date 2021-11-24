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
Test if we can determine the type of tumor for breast cancer data.
"""
# Extract data for ScikitLearn
cancer = load_breast_cancer()
data = cancer.data
target = cancer.target

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.25, random_state=seed)

y_train = y_train.reshape(1, -1)
y_test = y_test.reshape(1, -1)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train).T
X_test = scaler.transform(X_test).T

neural_network = NeuralNetwork(30, random_state=seed)
neural_network.add_layer(50, ReLU())
neural_network.add_layer(50, leakyReLU(alpha=0.5))
neural_network.add_layer(50, ReLU())
neural_network.add_layer(1, sigmoid())

epochs = 1000
size_batches = 5
regularization = 1e-2
learning_rate = lambda x: 0.01
neural_network.train(X_train, y_train, grad_cross_entropy, epochs, learning_rate, 
                     size_batches, regularization, input_test=X_test, target_test=y_test)

pred = neural_network.predict(X_test)
pred = pred.round()

print(f"accuracy: {np.sum(pred == y_test)}/{y_test.shape[1]} = {np.sum(pred == y_test)/y_test.shape[1]:.3f}")
