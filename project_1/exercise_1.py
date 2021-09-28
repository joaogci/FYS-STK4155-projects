import numpy as np
import numpy.linalg as linalg

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

from sklearn.model_selection import train_test_split

from functions import *

# parameters
degree = 5
a = 0
b = 1
n = 10
noise = 0.1

# random number generator
seed = 0
rng = np.random.default_rng(np.random.MT19937(seed=seed))

# generate x and y data
# the generation can be randomly distributed
x = np.linspace(a, b, n)
y = np.linspace(a, b, n)
# x = np.sort(rng.uniform(a, b, n))
# y = np.sort(rng.uniform(a, b, n))

# create a meshgrid and compute the franke function
x, y = np.meshgrid(x, y)
z = franke_function(x, y)

# add noise to the data
z += noise * rng.normal(0, 1, z.shape)

# ravel the data 
x_ravel = np.ravel(x).reshape((np.ravel(x).shape[0], 1))
y_ravel = np.ravel(y).reshape((np.ravel(y).shape[0], 1))
z_ravel = np.ravel(z).reshape((np.ravel(z).shape[0], 1))

# create the design matrix
X = create_X_2D(degree, x_ravel, y_ravel)

# train test split the data
X_train, X_test, z_train, z_test = train_test_split(X, z_ravel, test_size=0.25, random_state=seed)

# fit OLS model
betas = ols(X_train, z_train)
# predictions
z_pred = X_train @ betas
z_tilde = X_test @ betas

# scaling of the data (optional)
mean_X = np.mean(X_train, axis=0)
X_train_scaled = X_train - mean_X
X_test_scaled = X_test - mean_X

mean_z = np.mean(z_train)
z_train_scaled = z_train - mean_z
z_test_scaled = z_test - mean_z

# fit OLS model to franke function
betas_scaled = ols(X_train_scaled, z_train_scaled)
# predictions
z_pred_scaled = X_train_scaled @ betas_scaled
z_tilde_scaled = X_test_scaled @ betas_scaled

# compute the confidence interval of the betas
# we know that sgima^2 = diag(inv((X^T @ X)))
conf_interval_betas = np.sqrt(np.diag(linalg.pinv(X_train.T @ X_train)))
conf_interval_betas_scaled = np.sqrt(np.diag(linalg.pinv(X_train_scaled.T @ X_train_scaled)))


# prints
print(f"betas:        {betas.T}")
print(f"confidence interval: {conf_interval_betas.T}")
print()
print(f"betas_scaled: {betas_scaled.T}")
print(f"confidence interval: {conf_interval_betas_scaled.T}")
print()
print("Unscaled: ")
print(f"MSE train: {mean_squared_error(z_train, z_pred)} ")
print(f"R2 train: {r2_score(z_train, z_pred)} ")
print(f"MSE test: {mean_squared_error(z_test, z_tilde)} ")
print(f"R2 test: {r2_score(z_test, z_tilde)} ")
print("Scaled: ")
print(f"MSE train_scaled: {mean_squared_error(z_train_scaled, z_pred_scaled)} ")
print(f"R2 train_scaled: {r2_score(z_train_scaled, z_pred_scaled)} ")
print(f"MSE test_scaled: {mean_squared_error(z_test_scaled, z_tilde_scaled)} ")
print(f"R2 test_scaled: {r2_score(z_test_scaled, z_tilde_scaled)} ")


# confidence interval plots
plt.figure("confidence intervals for beta", figsize=(7, 9), dpi=80)

ax = plt.subplot(211)
plt.errorbar(np.arange(betas.shape[0]), betas, yerr=2*conf_interval_betas, fmt='-ob', capsize=4)
plt.title("unscaled")
plt.xlim((-1, betas.shape[0]+1))
plt.xlabel(r"$i$")
plt.ylabel(r"$\beta_i \pm 2\sigma$")

ax = plt.subplot(212)
plt.errorbar(np.arange(betas_scaled.shape[0]), betas_scaled, yerr=2*conf_interval_betas_scaled, fmt='-ob', capsize=4)
plt.xlim((-1, betas.shape[0]+1))
plt.title("scaled")
plt.xlabel(r"$i$")
plt.ylabel(r"$\beta_i \pm 2\sigma$")

plt.show()

# plot the franke function
fig = plt.figure("Franke Function", figsize=(8, 6), dpi=80)
ax = plt.axes(projection='3d')

# Plot the surface.
surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
# Customize the z axis.
ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

# plt.show()


