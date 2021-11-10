
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
n = 100
noise = 0.1
seed = 0
epochs = 100
n_nodes = 60
n_layers = 6
activation_fns = [Sigmoid(), ReLU(), ELU(), LeakyReLU(), Tanh()]
learning_rate = 0.00000005

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
mses = np.ndarray(len(activation_fns))
for j, activation_fn in enumerate(activation_fns):

    # Prepare network
    nn = Model(2, cost_function=cost_fn, random_state=seed)
    for i in range(0, n_layers):
        nn.add_layer(HiddenLayer(n_nodes, activation_function=activation_fn))
    nn.add_layer(OutputLayer(1, activation_function=Linear()))

    # Train network
    start = time()
    train_mse, test_mse = nn.train(X_train, z_train, learning_rate, sgd=False, epochs=epochs, testing_inputs=X_test, testing_targets=z_test, verbose=False)
    time_taken = time() - start
    
    # Save results as we go
    mses[j] = test_mse if not np.isnan(test_mse) else 0
    results['results'].append({
        'activation_fn': activation_fn.name(),
        'time': time_taken,
        'train_mse': train_mse,
        'test_mse': test_mse
    })

# save file
with open('results/activation_fn_comparison.pickle', 'wb') as handle:
    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('\n\nDumped results:\n', results)

# plot data
plt.figure()
plt.plot(range(len(activation_fns)), mses)
plt.xlabel('Activation function') # This is kind of a shit plot, we'd better actually display the name of these activation functions, but oh well - will do for now, and we can re-make this plot later from the results pickle later regardless
plt.ylabel('MSE')
if __name__ == "__main__":
    plt.show()
