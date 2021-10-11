import numpy as np
import matplotlib.pyplot as plt
from time import time

from functions import Regression


# parameters
max_degree = 15
n_vals = [400, 600, 800, 1000]
noise = 0.25
max_bootstrap = 100
degrees = np.arange(1, max_degree + 1)

# rng and seed
seed = 1963

# plot bias-variance trade-off
plt.figure("bias-variance trade-off", figsize=(9, 7))

# bootstrap for bias and var
for j, n in enumerate(n_vals):
    # regression object
    reg = Regression(max_degree, n, noise, seed)

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
    plt.ylabel(r"MSE")
    plt.legend()    

plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.95, 
                    top=0.95, 
                    wspace=0.25, 
                    hspace=0.25)

plt.savefig(f"./images/ex2_bias_var_bsc_{max_bootstrap}_noise_{noise}.pdf", dpi=400)

# plt.show()

