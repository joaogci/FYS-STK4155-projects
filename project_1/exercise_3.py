import numpy as np
import matplotlib.pyplot as plt
from time import time

from functions import Regression


# parameters
max_degree = 15
n = 800
noise = 0.2
max_bootstrap = 50
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

plt.show()


"""
# parameters
degree = 5
a = 0
b = 1
n = 50
noise = 0.2

# random number generator
seed = 0
rng = np.random.default_rng(np.random.MT19937(seed=seed))

# generate x and y data
x = rng.uniform(a, b, (n, 1))
y = rng.uniform(a, b, (n, 1))

# create a meshgrid and compute the franke function (with noise)
# x, y = np.meshgrid(x, y)
z = franke_function(x, y)
z += noise * rng.normal(0, 1, z.shape)

# ravel the data 
# x_ravel = np.ravel(x).reshape((np.ravel(x).shape[0], 1))
# y_ravel = np.ravel(y).reshape((np.ravel(y).shape[0], 1))
# z_ravel = np.ravel(z).reshape((np.ravel(z).shape[0], 1))

# design matrix, split and scaled
X = create_X_2D(degree, x, y)
# X_scaled, z_scaled = scale_mean_svd(X, z)

# Shuffle data
perm = rng.permuted(np.arange(0, x.shape[0]))
X_perm = X # X[perm]
z_perm = z # z[perm]

# X_scaled = X_scaled[perm]
# z_scaled = z_scaled[perm]

# k-folds
mse_overall = [None] * (11-5)
for kfolds in range(5, 11):
    kfold_size = np.floor(z.shape[0] / kfolds)
    mse = np.zeros(kfolds)
    for k in range(kfolds):
        
        # Split up into training/testing sets
        train_idx = np.concatenate((np.arange(0, k * kfold_size, dtype=int), np.arange((k+1) * kfold_size, kfolds * kfold_size, dtype=int)))
        test_idx = np.arange(k * kfold_size, (k+1) * kfold_size, dtype=int)

        X_train = X_perm[train_idx, :]
        z_train = z_perm[train_idx]
        X_test = X_perm[test_idx, :]
        z_test = z_perm[test_idx]

        # fit OLS model to franke function
        betas = ols(X_train, z_train)
        # predictions
        z_tilde = X_test @ betas

        mse[k] = mean_squared_error(z_test, z_tilde)

    print('K-folds:', kfolds, 'folds.\tAvg test MSE: %.3f' % (np.mean(mse)), '\tMin MSE: %.3f' % (np.min(mse)), '\tMax MSE: %.3f' % (np.max(mse)))
    mse_overall[kfolds-5] = mse

print()

plt.figure(1)
plt.boxplot(mse_overall, labels=range(5, 11))
plt.xlabel('# folds')
plt.ylabel('Mean Squared Error (OLS)')
plt.title('Franke Function K-Folds MSEs')
# plt.show()
"""
