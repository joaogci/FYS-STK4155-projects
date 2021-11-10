
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
epochs = 250
learning_rate = 1e-5
activation_fns = [Sigmoid(), LeakyReLU(alpha=5e-2), ELU(alpha=5e-2), Tanh(), ReLU()]

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

test_mses = []
train_mses = []
mse_lists = []
results = {'seed': seed, 'n':n, 'noise':noise, 'epochs':epochs, 'learning_rate':learning_rate, 'results':[]}
for i, activation_fn in enumerate(activation_fns):
    # Prepare network
    nn = Model(2, cost_function=cost_fn, random_state=seed)
    nn.add_layer(HiddenLayer(20, activation_function=Sigmoid()))
    nn.add_layer(HiddenLayer(20, activation_function=activation_fn))
    nn.add_layer(OutputLayer(1, activation_function=Linear()))

    # Train network
    start = time()
    train_mse, test_mse, mses = nn.train(X_train, z_train, learning_rate, sgd=True, epochs=epochs, testing_inputs=X_test, testing_targets=z_test, verbose=False, minibatch_size=5, return_errs=True)
    time_taken = time() - start

    # Write results
    test_mses.append(test_mse)
    train_mses.append(train_mse)
    mse_lists.append(mses)
    results['results'].append({
        'activation_fn': activation_fn.name(),
        'time': time_taken,
        'test_mse': test_mse,
        'train_mse': train_mse,
        'mse_by_epoch': mses
    })

# Save to file
with open('results/activation_comparison.pickle', 'wb') as handle:
    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('\n\nDumped results:\n', results)

plt.figure()
styles = ['--', '-', '--', '--', '.']
for i, mses in enumerate(mse_lists):
    plt.plot(range(0, len(mses), 5), mses[::5], styles[i], label=activation_fns[i].name(), alpha=0.9)
plt.legend()
plt.show()
