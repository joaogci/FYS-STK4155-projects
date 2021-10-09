import numpy as np
import matplotlib.pyplot as plt
from time import time

from functions import Regression


# parameters
max_degree = 15
degrees = np.arange(1, max_degree + 1)
n = 400
noise = 0.25
max_bootstrap = 100
n_folds = 7
lambdas = np.logspace(-5, 1, 50)
n_lambdas = lambdas.shape[0]

# min lmd and deg arrays
min_mse = np.zeros(2)
lmd_min = np.zeros(2)
deg_min = np.zeros(2)

# rng and seed
seed = int(time())
rng = np.random.default_rng(np.random.MT19937(seed=seed))
# regression object
reg = Regression(max_degree, n, noise, rng)


plt.figure(f"bootstrap vs cv", figsize=(9, 9))

# mse vs (lambdas, degs) for bootstrap
# bootstrap for MSE
mse = np.zeros((n_lambdas, max_degree))

for j, deg in enumerate(degrees):
    for i, lmd in enumerate(lambdas):
        mse[i, j], _, _ = reg.bootstrap(degree=deg, max_bootstrap_cycle=max_bootstrap, lmd=lmd)

min_mse_where = np.where(mse == np.min(mse))
lmd_min[0] = lambdas[min_mse_where[0][0]]
deg_min[0] = degrees[min_mse_where[1][0]]
min_mse[0] = mse[min_mse_where[0][0], min_mse_where[1][0]]


plt.contourf(np.log10(lambdas), degrees, mse.T)
plt.plot(np.log10(lambdas[min_mse_where[0][0]]), degrees[min_mse_where[1][0]], 'or')
plt.title(f"MSE for OLS with bootstrap with {max_bootstrap} cycles")
plt.ylabel(r"complexity")
plt.xlabel(r"\lambda")
plt.colorbar()

# mse vs (lambdas, degs) for cross validation
# cross validation for MSE
mse = np.zeros((n_lambdas, max_degree))

for j, deg in enumerate(degrees):
    for i, lmd in enumerate(lambdas):
        mse[i, j] = reg.k_folds_cross_validation(degree=deg, n_folds=n_folds, lmd=lmd)

min_mse_where = np.where(mse == np.min(mse))
lmd_min[1] = lambdas[min_mse_where[0][0]]
deg_min[1] = degrees[min_mse_where[1][0]]
min_mse[1] = mse[min_mse_where[0][0], min_mse_where[1][0]]

plt.subplot(1, 2, 2)

plt.contourf(np.log10(lambdas), degrees, mse.T)
plt.plot(np.log10(lambdas[min_mse_where[0][0]]), degrees[min_mse_where[1][0]], 'or')
plt.title(f"MSE for OLS with k-folds cross-validation with {n_folds} folds")
plt.ylabel(r"complexity")
plt.xlabel(r"\lambda")
plt.colorbar()

plt.savefig(f"./images/ex4_bs_bcs_{max_bootstrap}_cv_k_folds_{n_folds}_n_lmd_{n_lambdas}_n_{n}_noise_{noise}.pdf", dpi=400)

# plt.show()

# save min to file
with open("./ex4_min.txt", "w") as file:
    file.write("Bootstrap: \n")
    file.write(f"mse: {min_mse[0]}; lmd: {lmd_min[0]}; deg: {deg_min[0]} \n")
    
    file.write("Cross Validation: \n")
    file.write(f"mse: {min_mse[1]}; lmd: {lmd_min[1]}; deg: {deg_min[1]} \n")

