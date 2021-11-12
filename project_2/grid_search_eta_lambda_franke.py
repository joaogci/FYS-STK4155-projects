
import numpy as np
from scipy.sparse.construct import rand
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from time import time
import pickle
import seaborn as sns

from NeuralNetwork.Model import Model
from NeuralNetwork.Layer import HiddenLayer, OutputLayer
from NeuralNetwork.ActivationFunctions import Sigmoid, ReLU, ELU, LeakyReLU, Linear, Tanh, Softmax
from NeuralNetwork.cost_function.LinearRegression import LinearRegression

from functions import *


# params
n = 1000
noise = 0.1
seed = 0
epochs = 1000
n_nodes = 30
learning_rates = np.array([1e1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5])
reg_params = np.array([1e2, 1e1, 0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5])

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

# params
epochs = 500
size_batches = 5
eta_vals = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
reg_vals = [1e1, 0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]

mse_sig = np.zeros((len(eta_vals), len(reg_vals)))
mse_relu = np.zeros((len(eta_vals), len(reg_vals)))

for eta_i, eta in enumerate(eta_vals):
    for reg_i, regularization in enumerate(reg_vals):
        print()
        print(f"eta: {eta_i + 1}/{len(eta_vals)}; lmd: {reg_i + 1}/{len(reg_vals)}")
        
        nn = Model(2, cost_function=cost_fn, random_state=seed)
        nn.add_layer(HiddenLayer(n_nodes, activation_function=Sigmoid()))
        nn.add_layer(OutputLayer(1, activation_function=Linear()))
        nn.train(X_train, z_train, epochs=epochs, regularization=regularization, initial_learning_rate=eta)

        mse_sig[eta_i, reg_i] = nn.error(X_test, z_test)

        nn = Model(2, cost_function=cost_fn, random_state=seed)
        nn.add_layer(HiddenLayer(n_nodes, activation_function=ReLU()))
        nn.add_layer(OutputLayer(1, activation_function=Linear()))
        nn.train(X_train, z_train, epochs=epochs, regularization=regularization, initial_learning_rate=eta)
        
        mse_relu[eta_i, reg_i] = nn.error(X_test, z_test)

sns.set()

fig, ax = plt.subplots()
sns.heatmap(mse_sig, annot=True, ax=ax, cmap="viridis", cbar_kws={'label': 'MSE'})
ax.set_title("Training MSE for sigmoid")
ax.set_ylabel("$\eta$")
ax.set_xlabel("$\lambda$")
ax.set_xticklabels(reg_vals)
ax.set_yticklabels(eta_vals)
plt.savefig("./figs/part_b/1_grid_search_sigmoid.pdf", dpi=400)

fig, ax = plt.subplots()
sns.heatmap(mse_relu, annot=True, ax=ax, cmap="viridis", cbar_kws={'label': 'MSE'})
ax.set_title("Training MSE for ReLU")
ax.set_ylabel("$\eta$")
ax.set_xlabel("$\lambda$")
ax.set_xticklabels(reg_vals)
ax.set_yticklabels(eta_vals)
plt.savefig("./figs/part_b/2_grid_search_relu.pdf", dpi=400)
