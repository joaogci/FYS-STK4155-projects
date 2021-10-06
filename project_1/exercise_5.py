import numpy as np
import matplotlib.pyplot as plt
from time import time

from functions import Regression


# parameters
max_degree = 15
degrees = np.arange(1, max_degree + 1)
n = 800
noise = 0.25
max_bootstrap = n
n_folds_vals = np.array([5, 7, 10])
alphas = np.logspace(-6, 1, 100)
n_alphas = alphas.shape[0]

# min lmd and deg arrays
min_mse = np.zeros(n_folds_vals.shape[0] + 1)
lmd_min = np.zeros(n_folds_vals.shape[0] + 1)
deg_min = np.zeros(n_folds_vals.shape[0] + 1)

# rng and seed
seed = int(time())
rng = np.random.default_rng(np.random.MT19937(seed=seed))
# regression object
reg = Regression(max_degree, n, noise, rng)

# bootstrap for MSE
mse = np.zeros((n_alphas, max_degree))

for j, deg in enumerate(degrees):
    for i, alpha in enumerate(alphas):
        mse[i, j], _, _ = reg.bootstrap(degree=deg, max_bootstrap_cycle=max_bootstrap, alpha=alpha)

min_mse[0] = np.where(mse == np.min(mse))
lmd_min[0] = min_mse[0][0]
deg_min[0] = min_mse[1][0]

# mse vs (lambdas, degs) for bootstrap
plt.figure(f"bootstrap", figsize=(11, 9), dpi=80)
plt.contourf(np.log10(alphas), degrees, mse.T)
plt.plot(np.log10(alphas[lmd_min]), degrees[deg_min], 'or')
plt.ylabel("degrees",fontsize=14)
plt.xlabel("lambdas",fontsize=14)
plt.colorbar()

plt.savefig(f"./images/ex5_bootstrap_n_lmd_{n_alphas}_n_{n}_noise_{noise}.pdf", dpi=400)


for idx, n_folds in enumerate(n_folds_vals):
    # cross validation for MSE
    mse = np.zeros((n_alphas, max_degree))

    for j, deg in enumerate(degrees):
        for i, alpha in enumerate(alphas):
            mse[i, j] = reg.k_folds_cross_validation(degree=deg, n_folds=n_folds, alpha=alpha)

    min_mse[idx + 1] = np.where(mse == np.min(mse))
    lmd_min[idx + 1] = min_mse[0][0]
    deg_min[idx + 1] = min_mse[1][0]
    
    # mse vs (lambdas, degs) for cross validation
    plt.figure(f"cross validation", figsize=(11, 9), dpi=80)

    plt.contourf(np.log10(alphas), degrees, mse.T)
    plt.plot(np.log10(alphas[lmd_min]), degrees[deg_min], 'or')
    plt.ylabel("degrees",fontsize=14)
    plt.xlabel("lambdas",fontsize=14)
    plt.colorbar()

    plt.savefig(f"./images/ex5_cv_k_folds_{n_folds}_n_lmd_{n_alphas}_n_{n}_noise_{noise}.pdf", dpi=400)

# plt.show()

# save min to file
with open("./ex5_min.txt", "w") as file:
    file.write("Bootstrap: ")
    file.write(f"mse: {min_mse[0]}; lmd: {lmd_min[0]}; deg: {deg_min[0]} \n")
    
    file.write("Cross Validation:")
    for i, n_folds in enumerate(n_folds_vals):    
        file.write(f"n_folds: {n_folds}; mse: {min_mse[i + 1]}; lmd: {lmd_min[i + 1]}; deg: {deg_min[i + 1]} \n")


