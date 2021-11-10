
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from time import time
import pickle

from NeuralNetwork.Model import Model
from NeuralNetwork.Layer import HiddenLayer, OutputLayer
from NeuralNetwork.ActivationFunctions import Sigmoid, ReLU, ELU, LeakyReLU, Linear, Tanh, Softmax
from NeuralNetwork.cost_function.LinearRegression import LinearRegression

from functions import *


# params
n = 1000
noise = 0.25
seed = 1337
epochs = 100
learning_rate = 1e-5

# init data
rng = np.random.default_rng(np.random.MT19937(seed=seed))
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
cost_fn = LinearRegression(X_train, z_train, X_test, z_test)

# Prepare network
nn = Model(2, cost_function=cost_fn, random_state=seed)
nn.add_layer(HiddenLayer(20, activation_function=Sigmoid(), initial_bias=1))
nn.add_layer(HiddenLayer(20, activation_function=Sigmoid(), initial_bias=1))
nn.add_layer(OutputLayer(1, activation_function=Linear(), initial_bias=1))

nn2 = Model(2, cost_function=cost_fn, random_state=seed)
nn2.add_layer(HiddenLayer(20, activation_function=Sigmoid(), initial_bias=0))
nn2.add_layer(HiddenLayer(20, activation_function=Sigmoid(), initial_bias=0))
nn2.add_layer(OutputLayer(1, activation_function=Linear(), initial_bias=0))

# Train network
start = time()
train_mse, test_mse, mses = nn.train(X_train, z_train, learning_rate, sgd=True, epochs=epochs, testing_inputs=X_test, testing_targets=z_test, verbose=False, minibatch_size=5, return_errs=True)
time_taken = time() - start

train_mse2, test_mse2, mses2 = nn2.train(X_train, z_train, learning_rate, sgd=True, epochs=epochs, testing_inputs=X_test, testing_targets=z_test, verbose=False, minibatch_size=5, return_errs=True)

diff = []
diffB = []
for k in range(len(nn.layers)):
    layer = nn.layers[k]
    layer2 = nn2.layers[k]
    for i in range(layer._weights.shape[0]):
        for j in range(layer._weights.shape[1]):
            diff.append(layer._weights[i, j] - layer2._weights[i, j])
    for i in range(layer._biases.shape[0]):
        diffB.append(layer._biases[i] - layer2._biases[i])
plt.figure()
plt.plot(range(len(diff)), diff, '.')
plt.plot(range(len(diffB)), diffB, '+')

print(nn.layers[0]._weights - nn2.layers[0]._weights)

plt.figure()
plt.plot(range(len(mses)), mses)
plt.plot(range(len(mses2)), mses2)
plt.show()
