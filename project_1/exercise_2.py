import numpy as np
import numpy.linalg as linalg

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

from sklearn.model_selection import train_test_split
from sklearn.utils import resample

from time import time

from functions import create_X_2D, ols, franke_function, mean_squared_error, scale_mean


# parameters
max_degree = 15
max_bootstrap_cycle = 50
a = 0
b = 1
n = 800
noise = 0.2

# random number generator
seed = int(time())
rng = np.random.default_rng(np.random.MT19937(seed=seed))

# generate x and y data
# the generation can be randomly distributed
# x = np.linspace(a, b, (n, 1))
# y = np.linspace(a, b, (n, 1))
x = rng.uniform(a, b, (n, 1))
y = rng.uniform(a, b, (n, 1))

# compute the franke function
z = franke_function(x, y)

# add noise to the data
z += noise * rng.normal(0, 1, z.shape)

# mse[0, :] -> train
# mse[1, :] -> test
mse_train = np.zeros(max_degree)
mse_test = np.zeros(max_degree)
var = np.zeros(max_degree)
bias = np.zeros(max_degree)

# create the design matrix, split it and scale it
X_max_deg = create_X_2D(max_degree, x, y)
X_train_, X_test_, z_train, z_test = train_test_split(X_max_deg, z, test_size=0.25)
X_train_, X_test_, z_train, z_test = scale_mean(X_train_, X_test_, z_train, z_test)

max_features = lambda deg: int((deg) * (deg + 1) / 2)

for i, deg in enumerate(range(1, max_degree + 1)):
    # select the important features
    X_train = X_train_[:, :max_features(deg + 1)]
    X_test = X_test_[:, :max_features(deg + 1)]
    
    z_tilde_all = np.zeros((z_test.shape[0], max_bootstrap_cycle))
    
    # loop over bootstrap cycles
    for bootstrap_cycle in range(max_bootstrap_cycle):
        print(f"bootstrap cycle {bootstrap_cycle+1}/{max_bootstrap_cycle} with degree {deg}/{max_degree} ", end="\r")
        
        # split and scale the data
        X_train_resampled, z_train_resampled = resample(X_train, z_train)
        
        # fit OLS model to franke function
        betas = ols(X_train_resampled, z_train_resampled)
        # predictions
        z_pred = X_train_resampled @ betas
        z_tilde = X_test @ betas
        
        z_tilde_all[:, bootstrap_cycle] = z_tilde.reshape((z_tilde.shape[0], ))
        
        mse_train[i] += mean_squared_error(z_train_resampled, z_pred)
    
    # compute MSE, BIAS and VAR
    mse_train[i] /= max_bootstrap_cycle
    mse_test[i] = mean_squared_error(z_test, z_tilde_all)
    bias[i] = np.mean((z_test.reshape(z_test.shape[0], ) - np.mean(z_tilde_all, axis=1))**2)
    var[i] = np.mean(np.var(z_tilde_all, axis=1))

# plots

plt.figure("MSE vs complexity")
plt.plot(np.arange(1, max_degree + 1), mse_train, '-b', label='MSE train')
plt.plot(np.arange(1, max_degree + 1), mse_test, '-r', label='MSE test')
plt.xlabel(r"complexity")
plt.ylabel(r"$MSE$")
plt.legend()

plt.figure("MSE vs Bias^2 + Var")
plt.plot(np.arange(1, max_degree + 1), mse_test, '-r', label='MSE')
plt.plot(np.arange(1, max_degree + 1), var, '--g', label='var')
plt.plot(np.arange(1, max_degree + 1), bias, '--g', label='bias', alpha=0.35)
plt.plot(np.arange(1, max_degree + 1), bias + var, '--b', label='var+bias')
plt.xlabel(r"complexity")
plt.ylabel(r"$MSE$")
plt.legend()

plt.show()
    
        



