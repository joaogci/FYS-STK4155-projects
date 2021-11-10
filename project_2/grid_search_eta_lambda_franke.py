
import numpy as np
from scipy.sparse.construct import rand
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
noise = 0.1
seed = 0
epochs = 500
n_nodes = 30
activation_fn = Sigmoid()
learning_rate = 0.0005

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

# Train with grid search
nn = Model(2, cost_function=cost_fn, random_state=seed)
nn.add_layer(HiddenLayer(n_nodes, activation_function=activation_fn))
nn.add_layer(OutputLayer(1, activation_function=Linear()))
nn.grid_train(X_train, z_train, X_test, z_test, 'grid_search_eta_lambda_franke', plot=False, initial_learning_rate=np.logspace(-5, -1, 5), regularization=np.logspace(-4, 0, 5), epochs=epochs)
