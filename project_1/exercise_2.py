import numpy as np
import matplotlib.pyplot as plt
from time import time

from functions import Regression


# parameters
max_degree = 15
n_vals = [400, 600, 800, 1200]
noise = 0.2
max_bootstrap = 50
degrees = np.arange(1, max_degree + 1)

# rng and seed
seed = int(time())
rng = np.random.default_rng(np.random.MT19937(seed=seed))

# plot bias-variance trade-off
plt.figure("bias-variance trade-off", figsize=(11, 9), dpi=80)

# bootstrap for bias and var
for j, n in enumerate(n_vals):
    # regression object
    reg = Regression(max_degree, n, noise, rng)

    mse = np.zeros(max_degree)
    bias = np.zeros(max_degree)
    var = np.zeros(max_degree)
    
    for i, deg in enumerate(range(1, max_degree + 1)):
        mse[i], bias[i], var[i] = reg.bootstrap(degree=deg, max_bootstrap_cycle=max_bootstrap)

    plt.subplot(2, 2, j+1)
    
    plt.title(fr"$n={n}$")

    plt.plot(degrees, mse, '-r', label='MSE')
    plt.plot(degrees, var, '--k', label='var')
    plt.plot(degrees, bias, '--k', label='bias', alpha=0.40)
    plt.plot(degrees, bias + var, '-.k', label='var+bias')

    plt.xlabel(r"complexity")
    plt.legend()    

plt.show()

