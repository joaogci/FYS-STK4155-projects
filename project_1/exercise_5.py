import numpy as np
import matplotlib.pyplot as plt
from time import time

from functions import Regression


# parameters
max_degree = 15
degrees = np.arange(1, max_degree + 1)
n = 600
noise = 0.25
max_bootstrap = 100
n_folds = 7
alphas = np.logspace(-5, 1, 50)
n_alphas = alphas.shape[0]

# min lmd and deg arrays
min_mse = np.zeros(2)
lmd_min = np.zeros(2)
deg_min = np.zeros(2)

# rng and seed
seed = 1963
# regression object
reg = Regression(max_degree, n, noise, seed)


plt.figure(f"bootstrap", figsize=(11, 5))

# mse vs (lambdas, degs) for bootstrap
# bootstrap for MSE
mse = np.zeros((n_alphas, max_degree))

for j, deg in enumerate(degrees):
    for i, alpha in enumerate(alphas):
        mse[i, j], _, _ = reg.bootstrap(degree=deg, max_bootstrap_cycle=max_bootstrap, alpha=alpha)

min_mse_where = np.where(mse == np.min(mse))
lmd_min[0] = alphas[min_mse_where[0][0]]
deg_min[0] = degrees[min_mse_where[1][0]]
min_mse[0] = mse[min_mse_where[0][0], min_mse_where[1][0]]

plt.subplot(121)
plt.contourf(np.log10(alphas), degrees, mse.T)
plt.plot(np.log10(alphas[min_mse_where[0][0]]), degrees[min_mse_where[1][0]], 'or')
plt.title(f"MSE for Lasso with bootstrap with {max_bootstrap} cycles")
plt.ylabel(r"complexity")
plt.xlabel(r"$\log_{10}(\lambda)$")
plt.colorbar()


# regression object
reg = Regression(max_degree, n, noise, seed)

# mse vs (lambdas, degs) for cross validation
# cross validation for MSE
mse = np.zeros((n_alphas, max_degree))

for j, deg in enumerate(degrees):
    for i, alpha in enumerate(alphas):
        mse[i, j] = reg.k_folds_cross_validation(degree=deg, n_folds=n_folds, alpha=alpha)

min_mse_where = np.where(mse == np.min(mse))
lmd_min[1] = alphas[min_mse_where[0][0]]
deg_min[1] = degrees[min_mse_where[1][0]]
min_mse[1] = mse[min_mse_where[0][0], min_mse_where[1][0]]

plt.subplot(122)
plt.contourf(np.log10(alphas), degrees, mse.T)
plt.plot(np.log10(alphas[min_mse_where[0][0]]), degrees[min_mse_where[1][0]], 'or')
plt.title(f"MSE for Lasso with k-folds cross-validation with {n_folds} folds")
plt.ylabel(r"complexity")
plt.xlabel(r"$\log_{10}(\lambda)$")
plt.colorbar()

plt.subplots_adjust(left=0.05,
                    bottom=0.1, 
                    right=0.95, 
                    top=0.95, 
                    wspace=0.15, 
                    hspace=0.25)

plt.savefig(f"./images/ex5_bs_bcs_{max_bootstrap}_cv_k_folds_{n_folds}_n_lmd_{n_alphas}_n_{n}_noise_{noise}.pdf", dpi=400)
# plt.show()

# save min to file
with open("./ex5_min.txt", "w") as file:
    file.write("Bootstrap: \n")
    file.write(f"mse: {min_mse[0]}; lmd: {lmd_min[0]}; deg: {deg_min[0]} \n")
    
    file.write("Cross Validation: \n")
    file.write(f"mse: {min_mse[1]}; lmd: {lmd_min[1]}; deg: {deg_min[1]} \n")


