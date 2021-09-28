import numpy as np
import numpy.linalg as linalg

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

from sklearn.model_selection import train_test_split

from functions import create_X_2D, ols, franke_function, mean_squared_error, r2_score, scale_mean

# parameters
max_degree = 30
max_bootstrap_cycle = 50
a = 0
b = 1
n = 400
noise = 0.25

# random number generator
seed = 0
rng = np.random.default_rng(np.random.MT19937(seed=seed))

# generate x and y data
# the generation can be randomly distributed
x = np.linspace(a, b, n)
y = np.linspace(a, b, n)
# x = np.sort(rng.uniform(a, b, n))
# y = np.sort(rng.uniform(a, b, n))

# create a meshgrid and compute the franke function
x, y = np.meshgrid(x, y)
z = franke_function(x, y)

# add noise to the data
z += noise * rng.normal(0, 1, z.shape)

# ravel the data 
x_ravel = np.ravel(x).reshape((np.ravel(x).shape[0], 1))
y_ravel = np.ravel(y).reshape((np.ravel(y).shape[0], 1))
z_ravel = np.ravel(z).reshape((np.ravel(z).shape[0], 1))

# mse[0, ...] -> train 
# mse[1, ...] -> test
mse = np.zeros((2, max_degree - 1))
r2 = np.zeros((2, max_degree - 1))

for i, deg in enumerate(range(1, max_degree)):
    # create X, split and scale
    X = create_X_2D(deg, x_ravel, y_ravel)
    
    # loop over bootstrap cycles
    for bootstrap_cycle in range(max_bootstrap_cycle):
        # split and scale the data
        X_train, X_test, z_train, z_test = train_test_split(X, z_ravel, test_size=0.25)
        X_train_scaled, X_test_scaled, z_train_scaled, z_test_scaled = scale_mean(X_train, X_test, z_train, z_test)
        
        # fit OLS model to franke function
        betas = ols(X_train_scaled, z_train_scaled)
        # predictions
        z_pred = X_train_scaled @ betas
        z_tilde = X_test_scaled @ betas
        
        # compute MSE and R2 score
        mse[0, i] += mean_squared_error(z_train_scaled, z_pred)
        mse[1, i] += mean_squared_error(z_test_scaled, z_tilde)
        
        r2[0, i] += r2_score(z_train_scaled, z_pred)
        r2[1, i] += r2_score(z_test_scaled, z_tilde)


