import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection

from NeuralNetwork.Model import Model
from NeuralNetwork.Layer import Layer, HiddenLayer, OutputLayer
from NeuralNetwork.activation.Sigmoid import Sigmoid
from NeuralNetwork.activation.Linear import Linear

from functions import *

from sklearn.model_selection import train_test_split

# params
n = 100
noise = 0
seed = 1337
iterations = 1000

# rng 
rng = np.random.default_rng(np.random.MT19937(seed=seed))

# data
x = np.sort(rng.uniform(0, 1, int(np.sqrt(n))))
y = np.sort(rng.uniform(0, 1, int(np.sqrt(n))))

x, y = np.meshgrid(x, y)

z = franke_function(x, y)
z += noise * rng.normal(0, 1, z.shape)

x = np.ravel(x)
y = np.ravel(y)
z = np.ravel(z)

X = np.zeros((x.shape[0], 2))
X[:, 0] = x
X[:, 1] = y

X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)
z_train, z_test = np.matrix(z_train).T, np.matrix(z_test).T

n_data_points = X_train.shape[0]

# input shape = (n, 2)   |   output shape = (n, 1)
neural_network = Model(2, random_state=seed)
neural_network.add_layer(HiddenLayer(100, Sigmoid()))
neural_network.add_layer(HiddenLayer(100, Sigmoid()))
neural_network.add_layer(OutputLayer(1, Linear()))

outputs = neural_network.feed_forward(X_train)

print()
for i in range(iterations):
    print(int(i / iterations * 100), '%', end='\r')
    neural_network.back_prop(X_train, z_train, learning_rate=0.02)
print('100%')

# Print final outputs
print("Train data: ")
print('\nAfter training:')
print(neural_network.feed_forward(X_train))
print('\nTargets:')
print(z_train)
print('\nMSE:', neural_network.fwd_mse(X_train, z_train))

print("\n\nTest data: ")
print('\nAfter training:')
print(neural_network.feed_forward(X_test))
print('\nTargets:')
print(z_test)
print('\nMSE:', neural_network.fwd_mse(X_test, z_test))