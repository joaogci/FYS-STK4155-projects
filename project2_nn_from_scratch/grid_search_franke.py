import sys
sys.path.append('../')

import numpy as np
from Network import NeuralNetwork

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from Activation import *
from Cost import *

import seaborn as sns
import matplotlib.pyplot as plt

from functions import * 

seed = 1337

sns.set_theme(style="white")

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

# params
epochs = 1
size_batches = 5
eta_vals = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
reg_vals = [0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]

mse = np.zeros((len(eta_vals), len(reg_vals)))

for eta_i, eta in enumerate(eta_vals):
    for reg_i, regularization in enumerate(reg_vals):
        print()
        print(f"eta: {eta_i + 1}/{len(eta_vals)}; batch: {reg_i + 1}/{len(reg_vals)}")
        
        neural_network = NeuralNetwork(2, random_state=seed)
        neural_network.add_layer(20, sigmoid())
        neural_network.add_layer(20, sigmoid())
        neural_network.add_layer(1, linear())

        learning_rate = lambda x: eta
        neural_network.train(X_train, y_train, grad_mse, epochs, learning_rate, size_batches, regularization)

        pred = neural_network.predict(X_test)
        # pred = pred.round()
        mse[eta_i, reg_i] = mean_squared_error(y_test, pred)

sns.set()

fig, ax = plt.subplots()
sns.heatmap(mse, annot=True, ax=ax, cmap="viridis")
ax.set_title("Training MSE")
ax.set_ylabel("$\eta$")
ax.set_xlabel("$\lambda$")
ax.set_xticklabels(reg_vals)
ax.set_yticklabels(eta_vals)
plt.show()

