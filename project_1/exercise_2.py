import numpy as np
import numpy.linalg as linalg

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

from sklearn.model_selection import train_test_split
from sklearn.utils import resample

from functions import create_X_2D, ols, franke_function, mean_squared_error, r2_score, scale_mean, bias_squared, variance

# parameters
max_degree = 15
max_bootstrap_cycle = 50
a = 0
b = 1
n = 1000
noise = 0.2

# random number generator
seed = 0
rng = np.random.default_rng(np.random.MT19937(seed=seed))

# generate x and y data
# the generation can be randomly distributed
# x = np.linspace(a, b, n)
# y = np.linspace(a, b, n)
x = np.sort(rng.uniform(a, b, int(np.sqrt(n))))
y = np.sort(rng.uniform(a, b, int(np.sqrt(n))))

# create a meshgrid and compute the franke function
x, y = np.meshgrid(x, y)
z = franke_function(x, y)

# add noise to the data
z += noise * rng.normal(0, 1, z.shape)

# ravel the data 
x_ravel = np.ravel(x).reshape((np.ravel(x).shape[0], 1))
y_ravel = np.ravel(y).reshape((np.ravel(y).shape[0], 1))
z_ravel = np.ravel(z).reshape((np.ravel(z).shape[0], 1))

# mse[0, :] -> train
# mse[1, :] -> test
mse_train = np.zeros(max_degree)
mse_test = np.zeros(max_degree)
var = np.zeros(max_degree)
bias = np.zeros(max_degree)

for i, deg in enumerate(range(1, max_degree + 1)):
    # create X, split and scale
    X = create_X_2D(deg, x_ravel, y_ravel)
    
    # loop over bootstrap cycles
    for bootstrap_cycle in range(max_bootstrap_cycle):
        print(f"bootstrap cycle {bootstrap_cycle+1}/{max_bootstrap_cycle} with degree {deg}/{max_degree} ", end="\r")
        
        # split and scale the data
        X_train, X_test, z_train, z_test = train_test_split(X, z_ravel, test_size=0.25)
        X_train, z_train = resample(X_train, z_train)
        X_train, X_test, z_train, z_test = scale_mean(X_train, X_test, z_train, z_test)
        
        # fit OLS model to franke function
        betas = ols(X_train, z_train)
        # predictions
        z_pred = X_train @ betas
        z_tilde = X_test @ betas
        
        # compute MSE and R2 score
        mse_train[i] += mean_squared_error(z_train, z_pred)
        mse_test[i] += mean_squared_error(z_test, z_tilde)
        
        var[i] += variance(z_tilde)
        
        bias[i] += bias_squared(z_test, z_tilde)
        
mse_train /= max_bootstrap_cycle
mse_test /= max_bootstrap_cycle 
var /= max_bootstrap_cycle
bias /= max_bootstrap_cycle

plt.figure("MSE vs complexity")
plt.plot(np.arange(1, max_degree + 1), mse_train, '-b', label='MSE train')
plt.plot(np.arange(1, max_degree + 1), mse_test, '-r', label='MSE test')
plt.ylim((0, np.max(mse_test) + 0.20))
plt.xlabel(r"$degree$")
plt.ylabel(r"$MSE$")
plt.legend()

plt.figure("MSE vs Bias^2 + Var")
plt.plot(np.arange(1, max_degree + 1), mse_test, '-r', label='MSE')
plt.plot(np.arange(1, max_degree + 1), var, '--g', label='var')
plt.plot(np.arange(1, max_degree + 1), bias, '--g', label='bias', alpha=0.35)
plt.plot(np.arange(1, max_degree + 1), bias + var, '--b', label='var+bias')
plt.ylim((0, np.max(bias + var) + 0.10))
plt.xlabel(r"$degree$")
plt.ylabel(r"$MSE$")
plt.legend()


plt.show()
    
        



