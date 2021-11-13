
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
noise = 0.1
seed = 0
epochs = 500
n_nodes = 400
max_n_layers = 20 # 1 layer of 1000 nodes up to 20 layers of 50 nodes each
activation_fn = Sigmoid()
learning_rate = 0.001

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

# run neural network
results = {'seed': seed, 'results': []}
mses = np.ndarray(max_n_layers)
for j, n_layers in enumerate(range(1, max_n_layers+1)):

    # Prepare network
    n_nodes_per_layer = 20
    nn = Model(2, cost_function=cost_fn, random_state=seed)
    for i in range(0, n_layers):
        nn.add_layer(HiddenLayer(n_nodes_per_layer, activation_function=activation_fn))
    nn.add_layer(OutputLayer(1, activation_function=Linear()))

    # Train network
    start = time()
    train_mse, test_mse = nn.train(X_train, z_train, learning_rate, sgd=False, epochs=epochs, testing_inputs=X_test, testing_targets=z_test, verbose=False)
    time_taken = time() - start
    
    # Save results as we go
    mses[j] = test_mse
    results['results'].append({
        'n_layers': n_layers,
        'n_nodes_per_layer': n_nodes_per_layer,
        'time': time_taken,
        'train_mse': train_mse,
        'test_mse': test_mse
    })

# save file
with open('results/layers_comparison.pickle', 'wb') as handle:
    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('\n\nDumped results:\n', results)

# plot data
plt.figure()
plt.plot(range(1, max_n_layers+1), mses)
plt.xlabel('number of hidden layers')
plt.ylabel('MSE')
plt.title(f"Testing MSE {epochs} epochs, 20 nodes per layer")

plt.savefig("./figs/part_c/3_mse_layers.pdf", dpi=400)
