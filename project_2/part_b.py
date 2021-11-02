
import numpy as np
import matplotlib.pyplot as plt
from functions import *

from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

from NeuralNetwork.Model import Model
from NeuralNetwork.Layer import *
from NeuralNetwork.activation.Sigmoid import Sigmoid
from NeuralNetwork.cost_function.LinearRegression import LinearRegression
from NeuralNetwork.StochasticGradientDescent import StochasticGradientDescent


# params
n = 400
noise = 0.25
seed = 10

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

lin_reg = LinearRegression(X, z)
optimizer = StochasticGradientDescent(lin_reg, size_minibatches=10, t0=50, t1=2, rng=rng)
theta_hat = optimizer.optimize()

theta_ols = ols(X, z)

print("theta_ols:", theta_ols)
print("theta_sgd:", theta_hat)
