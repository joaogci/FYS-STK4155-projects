import numpy as np
import matplotlib.pyplot as plt
from time import time

from functions import Regression


# parameters
max_degree = 15
degrees = np.arange(1, max_degree + 1)
n = 400
noise = 0.2
max_bootstrap = 15
n_folds = 5
lambdas = np.logspace(-4, 3, 30)
n_lambdas = lambdas.shape[0]

# rng and seed
seed = int(time())
rng = np.random.default_rng(np.random.MT19937(seed=seed))
# regression object
reg = Regression(max_degree, n, noise, rng)

"""
mse = np.zeros((n_lambdas, max_degree))
bias = np.zeros((n_lambdas, max_degree))
var = np.zeros((n_lambdas, max_degree))

# bootstrap for bias and var
for j, deg in enumerate(degrees):
    for i, lmd in enumerate(lambdas):
        mse[i, j], bias[i, j], var[i, j] = reg.bootstrap(degree=deg, max_bootstrap_cycle=max_bootstrap, lmd=lmd)

# plot bias-variance trade-off
# plt.figure(1, figsize=(11, 9), dpi=80)

# L, D = np.meshgrid(lambdas, degrees)
# plt.contour(lambdas, degrees, mse)
# plt.ylabel("degrees")
# plt.xlabel("lambdas")

plt.figure(2, figsize=(11, 9), dpi=80)

L, D = np.meshgrid(degrees, lambdas)
plt.contourf(np.log10(lambdas), degrees, mse.T)
plt.ylabel("degrees",fontsize=14)
plt.xlabel("lambdas",fontsize=14)
plt.colorbar()
"""
# plt.show()


# plot bias-variance trade-off
# plt.figure("k-folds cross validation for ridge", figsize=(11, 9), dpi=80)

# cross validation
mse = np.zeros((n_lambdas, max_degree))

for j, deg in enumerate(degrees):
    for i, lmd in enumerate(lambdas):
        mse[i, j] = reg.k_folds_cross_validation(degree=deg, n_folds=5, lmd=lmd)

plt.figure(2, figsize=(11, 9), dpi=80)

L, D = np.meshgrid(degrees, lambdas)
plt.contourf(np.log10(lambdas), degrees, mse.T)
plt.ylabel("degrees",fontsize=14)
plt.xlabel("lambdas",fontsize=14)
plt.colorbar()

plt.show()

