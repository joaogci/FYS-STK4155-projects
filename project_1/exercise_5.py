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
alphas = np.logspace(-4, 3, 10)
n_alphas = alphas.shape[0]

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

print()

min_mse = np.where(mse == np.min(mse))
lmd_min = min_mse[0][0]
deg_min = min_mse[1][0]

# mse vs (lambdas, degs) for bootstrap
plt.figure(f"bootstrap; min[(lambda, deg)] = ({alphas[lmd_min]:.4f}, {degrees[deg_min]}), with mse={np.min(mse):.4f}", figsize=(11, 9), dpi=80)
plt.contourf(np.log10(alphas), degrees, mse.T)
plt.plot(np.log10(alphas[lmd_min]), degrees[deg_min], 'or')
plt.ylabel("degrees",fontsize=14)
plt.xlabel("lambdas",fontsize=14)
plt.colorbar()

plt.savefig(f"ex5_bootstrap_btc_{max_bootstrap}_n_{n}_noise_{noise}.pdf", dpi=400)

# cross validation for MSE
mse = np.zeros((n_alphas, max_degree))

for j, deg in enumerate(degrees):
    for i, alpha in enumerate(alphas):
        mse[i, j] = reg.k_folds_cross_validation(degree=deg, n_folds=5, alpha=alpha)

min_mse = np.where(mse == np.min(mse))
lmd_min = min_mse[0][0]
deg_min = min_mse[1][0]

# mse vs (lambdas, degs) for cross validation
plt.figure(f"cross validation; min[(lambda, deg)] = ({alphas[lmd_min]:.4f}, {degrees[deg_min]}), with mse={np.min(mse):.4f}", figsize=(11, 9), dpi=80)
plt.contourf(np.log10(alphas), degrees, mse.T)
plt.plot(np.log10(alphas[lmd_min]), degrees[deg_min], 'or')
plt.ylabel("degrees",fontsize=14)
plt.xlabel("lambdas",fontsize=14)
plt.colorbar()

plt.savefig(f"ex5_cv_k_folds_{n_folds}_n_lmd_{n_alphas}_n_{n}_noise_{noise}.pdf", dpi=400)

# plt.show()

