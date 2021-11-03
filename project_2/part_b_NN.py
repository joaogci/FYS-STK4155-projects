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
n = 1000
noise = 0
seed = 1337

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

X = create_X_2D(5, x, y)

X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)

n_data_points = X_train.shape[0]
n_features = X_train.shape[1]

neural_network = Model(n_features, random_state=seed)
neural_network.add_layer(HiddenLayer(100, Sigmoid()))
neural_network.add_layer(HiddenLayer(100, Sigmoid()))
neural_network.add_layer(OutputLayer(n_features, Linear()))

outputs = neural_network.feed_forward(X_train)

print(outputs)

print()
for i in range(1000):
    print(int(i / 1000 * 100), '%', end='\r')
    for j in range(len(z_train)): # For now, back_prop only deals with these one row at a time
        neural_network.back_prop(X_train[j], z_train[j], learning_rate=1)
print('100%')

# Print final outputs
print("Train data: ")
print('\nAfter training:')
print(neural_network.feed_forward(X_train))
print('\nTargets:')
print(z_train)

print("Test data: ")
print('\nAfter training:')
print(neural_network.feed_forward(X_test))
print('\nTargets:')
print(z_test)
