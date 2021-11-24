import sys
sys.path.append('../')

import numpy as np
from Network import NeuralNetwork

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from Activation import *
from Cost import *

from functions import franke_function

seed = 1337

"""
Test if we can determine the type of tumor for breast cancer data.
"""

noise = 0.25
n = 1000

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

X = np.array([x, y]).T

X_train, X_test, y_train, y_test = train_test_split(X, z, test_size=0.25, random_state=seed)

scaler = StandardScaler(with_std=False)
scaler.fit(X_train)
X_train = scaler.transform(X_train).T
X_test = scaler.transform(X_test).T

y_train = y_train.reshape(1, -1)
y_test = y_test.reshape(1, -1)

neural_network = NeuralNetwork(2, random_state=seed)
neural_network.add_layer(20, sigmoid())
neural_network.add_layer(20, sigmoid())
neural_network.add_layer(1, linear())

epochs = 500
size_batches = 5
regularization = 1e-2
learning_rate = lambda x: 0.0001
neural_network.train(X_train, y_train, grad_mse, epochs, learning_rate, size_batches, regularization, X_test, y_test)

pred = neural_network.predict(X_test)

print(f"error: {mse(y_test, pred)}")
