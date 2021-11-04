
import numpy as np
import matplotlib.pyplot as plt
from functions import *

from NeuralNetwork.Model import Model
from NeuralNetwork.Layer import *
from NeuralNetwork.activation_function.Sigmoid import Sigmoid
from NeuralNetwork.cost_function.LinearRegression import LinearRegression
from NeuralNetwork.optimizer.StochasticGradientDescent import StochasticGradientDescent
from NeuralNetwork.optimizer.GradientDescent import GradientDescent
from NeuralNetwork.optimizer.RMSprop import RMSprop

from sklearn.linear_model import SGDRegressor
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

lin_reg = LinearRegression(X_train, z_train, X_test, z_test)

optimizer_SGD = StochasticGradientDescent(lin_reg, size_minibatches=5, rng=rng)
theta_SGD = optimizer_SGD.optimize(iter_max=int(1e3), eta=0.01)

optimizer_RMS = RMSprop(lin_reg)
theta_RMS = optimizer_RMS.optimize(iter_max=int(1e3), eta=0.01)

theta_ols = ols(X_train, z_train)

print("theta_ols:", theta_ols)
print("theta_SGD:", theta_SGD)
print("theta_RMS:", theta_RMS)

print("MSE OLS:", mean_squared_error(z_test, X_test @ theta_ols))
print("MSE SGD:", mean_squared_error(z_test, X_test @ theta_SGD))
print("MSE RMS:", mean_squared_error(z_test, X_test @ theta_RMS))

plot_prediction_3D(theta_SGD, 5)
# plot_prediction_3D(theta_ols, 5)
