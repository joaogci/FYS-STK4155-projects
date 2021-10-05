import numpy as np
import matplotlib.pyplot as plt
from time import time

from functions import Regression


# parameters
max_degree = 15
n = 800
noise = 0.25
max_bootstrap = 100
n_folds_vals = [5, 7, 10]
degrees = np.arange(1, max_degree + 1)

# rng and seed
seed = int(time())
rng = np.random.default_rng(np.random.MT19937(seed=seed))

# figure plot
plt.figure("MSE comparison", figsize=(11, 9), dpi=80)

# cross validation
for j, n_folds in enumerate(n_folds_vals):
    # regression object
    reg = Regression(max_degree, n, noise, rng)

    mse_cv = np.zeros(max_degree)
    for i, deg in enumerate(degrees):
        mse_cv[i] = reg.k_folds_cross_validation(degree=deg, n_folds=n_folds)
    
    plt.subplot(2, 2, j+1)
    
    plt.plot(degrees, mse_cv, '-k')
    plt.xlabel(r"complexity")
    plt.ylabel(r"MSE")
    plt.title(f"k-folds cross validation with k={n_folds}")

# bootstrap
# regression object
reg = Regression(max_degree, n, noise, rng)

mse = np.zeros(max_degree)
bias = np.zeros(max_degree)
var = np.zeros(max_degree)

for i, deg in enumerate(range(1, max_degree + 1)):
    mse[i], bias[i], var[i] = reg.bootstrap(degree=deg, max_bootstrap_cycle=max_bootstrap)

plt.subplot(2, 2, 4)
plt.plot(degrees, mse, '-r')
plt.xlabel(r"complexity")
plt.ylabel(r"MSE")
plt.title(f"bootstrap with n_cycles={max_bootstrap}")

plt.savefig(f"./images/ex3_cv_bs_n_{n}_noise_{noise}.pdf", dpi=400)

# plt.show()

