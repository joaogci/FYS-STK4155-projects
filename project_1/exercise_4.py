import numpy as np
import numpy.linalg as linalg

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.utils import resample

from functions import create_X_2D, ridge, franke_function, mean_squared_error, r2_score, scale_mean_svd, bias_squared, variance

# parameters
degree = 5
max_bootstrap_cycle = 15
kfolds = 5
a = 0
b = 1
n = 50
noise = 0.2
λ_min = -12
λ_max = 2
λ_steps = 2

# random number generator
seed = 0
rng = np.random.default_rng(np.random.MT19937(seed=seed))

# generate x and y data
x = np.sort(rng.uniform(a, b, n))
y = np.sort(rng.uniform(a, b, n))

# create a meshgrid and compute the franke function + noise
x, y = np.meshgrid(x, y)
z = franke_function(x, y)
z += noise * rng.normal(0, 1, z.shape)

# ravel the data 
x_ravel = np.ravel(x).reshape((np.ravel(x).shape[0], 1))
y_ravel = np.ravel(y).reshape((np.ravel(y).shape[0], 1))
z_ravel = np.ravel(z).reshape((np.ravel(z).shape[0], 1))

# design matrix, split and scaled
X = create_X_2D(degree, x_ravel, y_ravel)
X_scaled, z_scaled = scale_mean_svd(X, z_ravel)

# Shuffle data
perm = rng.permuted(np.arange(0, x.shape[0]))
X_scaled = X_scaled[perm]
z_scaled = z_scaled[perm]

# Ridge regression over various values of λ
λ_count = (λ_max - λ_min) * λ_steps
mse_b = np.zeros(λ_count)
mse_k = np.zeros(λ_count)
lambdas = np.logspace(λ_min, λ_max, λ_count)
for i, λ in enumerate(lambdas):

    # loop over bootstrap cycles
    bootstrap_mse = np.zeros(max_bootstrap_cycle)
    for bootstrap_cycle in range(max_bootstrap_cycle):

        # split the data & resample
        X_train, X_test, z_train, z_test = train_test_split(X_scaled, z_scaled, test_size=0.25)
        X_train, z_train = resample(X_train, z_train)
        
        # fit OLS model to franke function
        betas = ridge(X_train, z_train, λ)
        # prediction
        z_tilde = X_test @ betas

        bootstrap_mse[bootstrap_cycle] = mean_squared_error(z_test, z_tilde)
    mse_b[i] = np.mean(bootstrap_mse)

    # K-Folds CV
    kfold_size = np.floor(z_scaled.shape[0] / kfolds)
    kfold_mse = np.zeros(kfolds)
    for k in range(kfolds):

        # Split up into training/testing sets
        train_idx = np.concatenate((np.arange(0, k * kfold_size, dtype=int), np.arange((k+1) * kfold_size, kfolds * kfold_size, dtype=int)))
        test_idx = np.arange(k * kfold_size, (k+1) * kfold_size, dtype=int)
        X_train = X_scaled[train_idx]
        z_train = z_scaled[train_idx]
        X_test = X_scaled[test_idx]
        z_test = z_scaled[test_idx]

        # fit OLS model to franke function
        betas = ridge(X_train, z_train, λ)
        # prediction
        z_tilde = X_test @ betas

        kfold_mse[k] = mean_squared_error(z_test, z_tilde)
    mse_k[i] = np.mean(kfold_mse)

plt.errorbar(np.log10(lambdas), mse_b, label='Bootstrap MSEs')
plt.errorbar(np.log10(lambdas), mse_k, label='K-Fold MSE')
plt.title('Bootstrap & K-Fold MSE over various λ values in ridge regression on the Franke Function')
plt.legend()
plt.xlabel('log10(λ)')
plt.ylabel('MSE')
plt.show()
