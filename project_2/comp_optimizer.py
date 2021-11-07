from types import LambdaType
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter, time

from scipy.sparse.construct import random

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

eta_vals = np.power(10.0, [-5, -4, -3, -2, -1])
eta = 0.001

lin_reg = LinearRegression(X_train, z_train, X_test, z_test)
time_GD = perf_counter()
optimizer_GD = GradientDescent(lin_reg)
GD_out = optimizer_GD.optimize(tol=1e-6, iter_max=int(1e6), eta=eta, random_state=seed, verbose=True)
time_GD = perf_counter() - time_GD

lin_reg = LinearRegression(X_train, z_train, X_test, z_test)
time_SGD = perf_counter()
optimizer_SGD = StochasticGradientDescent(lin_reg, size_minibatches=20)
SGD_out = optimizer_SGD.optimize(tol=1e-6, iter_max=int(1e6), eta=eta, random_state=seed, verbose=True)
time_SGD = perf_counter() - time_SGD

lin_reg = LinearRegression(X_train, z_train, X_test, z_test)
time_RMS = perf_counter()
optimizer_RMS = RMSprop(lin_reg)
RMS_out = optimizer_RMS.optimize(tol=1e-6, iter_max=int(1e6), eta=eta, random_state=seed, verbose=True)
time_RMS = perf_counter() - time_RMS

lin_reg = LinearRegression(X_train, z_train, X_test, z_test)
time_newton = perf_counter()
optimizer_newton = NewtonMethod(lin_reg)
newton_out = optimizer_newton.optimize(tol=1e-6, iter_max=int(1e6), eta=eta, random_state=seed, verbose=True)
time_newton = perf_counter() - time_newton

theta_ols = ols(X_train, z_train)

print()
print("epoch GD:", GD_out[1])
print("epoch SGD:", SGD_out[1])
print("epoch RMS:", RMS_out[1])
print("epoch newton:", newton_out[1])
print()
print("MSE OLS:", mean_squared_error(z_test, X_test @ theta_ols))
print("MSE GD:", mean_squared_error(z_test, X_test @ GD_out[0]), "; time:", time_GD)
print("MSE SGD:", mean_squared_error(z_test, X_test @ SGD_out[0]), "; time:", time_SGD)
print("MSE RMS:", mean_squared_error(z_test, X_test @ RMS_out[0]), "; time:", time_RMS)
print("MSE newton:", mean_squared_error(z_test, X_test @ newton_out[0]), "; time:", time_newton)

mse_ols = mean_squared_error(z_train, X_train @ theta_ols)

plt.figure("MSE vs epochs")

plt.loglog(np.arange(1, GD_out[1] + 2), GD_out[2], label=f'GD; eta={eta}')
plt.loglog(np.arange(1, SGD_out[1] + 2), SGD_out[2], label=f'SGD; eta={eta}')
plt.loglog(np.arange(1, RMS_out[1] + 2), RMS_out[2], label=f'RMSprop; eta={eta}')
plt.loglog(np.arange(1, newton_out[1] + 2), newton_out[2], label='Newton\'s method')
plt.loglog([1, RMS_out[1] + 2], [mse_ols, mse_ols], label='OLS')

plt.legend()


plt.show()


