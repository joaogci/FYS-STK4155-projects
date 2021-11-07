import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter

from NeuralNetwork.optimizer.GradientDescent import GradientDescent
from NeuralNetwork.optimizer.StochasticGradientDescent import StochasticGradientDescent
from NeuralNetwork.optimizer.RMSprop import RMSprop
from NeuralNetwork.optimizer.NewtonMethod import NewtonMethod

from NeuralNetwork.cost_function.LinearRegression import LinearRegression
from NeuralNetwork.cost_function.LogisticRegression import LogisticRegression

from functions import *

from sklearn.model_selection import train_test_split

# params
n = 10000
deg = 5
noise = 0.1
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

X = create_X_2D(deg, x, y)

X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.25, random_state=seed)

eta = 0.001
time = list()
mse = list()

for i, batch_size in enumerate([1, 2, 3, 4, 5, 10, 100, 1000]):

    lin_reg = LinearRegression(X_train, z_train, X_test, z_test)
    time_SGD = perf_counter()
    optimizer_SGD = StochasticGradientDescent(lin_reg, size_minibatches=batch_size)
    SGD_out = optimizer_SGD.optimize(tol=1e-6, iter_max=int(1e6), eta=eta, random_state=seed, verbose=True)
    time_SGD = perf_counter() - time_SGD
    
    time.append(time_SGD)
    mse.append(SGD_out[2][-1])
    
print(mse)
print(time)
    
