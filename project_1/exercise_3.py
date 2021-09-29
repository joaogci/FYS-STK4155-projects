import numpy as np

import matplotlib.pyplot as plt

from functions import create_X_2D, ols, franke_function, mean_squared_error, scale_mean_svd, bias_squared, variance

# parameters
degree = 6
a = 0
b = 1
n = 500
noise = 0.2

# random number generator
seed = 0
rng = np.random.default_rng(np.random.MT19937(seed=seed))

# generate x and y data
x = np.sort(rng.uniform(a, b, n))
y = np.sort(rng.uniform(a, b, n))

# create a meshgrid and compute the franke function (with noise)
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

# k-folds
mse_overall = [None] * (11-5)
for kfolds in range(5, 11):
    kfold_size = np.floor(z_scaled.shape[0] / kfolds)
    mse = np.zeros(kfolds)
    for k in range(kfolds):
        
        # Split up into training/testing sets
        train_idx = np.concatenate((np.arange(0, k * kfold_size, dtype=int), np.arange((k+1) * kfold_size, kfolds * kfold_size, dtype=int)))
        test_idx = np.arange(k * kfold_size, (k+1) * kfold_size, dtype=int)
        X_train = X_scaled[train_idx]
        z_train = z_scaled[train_idx]
        X_test = X_scaled[test_idx]
        z_test = z_scaled[test_idx]

        # fit OLS model to franke function
        betas = ols(X_train, z_train)
        # predictions
        z_tilde = X_test @ betas

        mse[k] = mean_squared_error(z_test, z_tilde)

    print('K-folds:', kfolds, 'folds.\tAvg test MSE: %.3f' % (np.mean(mse)), '\tMin MSE: %.3f' % (np.min(mse)), '\tMax MSE: %.3f' % (np.max(mse)))
    mse_overall[kfolds-5] = mse

plt.boxplot(mse_overall, labels=range(5, 11))
plt.xlabel('# folds')
plt.ylabel('Mean Squared Error (OLS)')
plt.title('Franke Function K-Folds MSEs')
plt.show()
