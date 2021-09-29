import numpy as np
import matplotlib.pyplot as plt
from time import time

from functions import Regression


# parameters
max_degree = 15
n = 800
noise = 0.2

# rng and seed
seed = int(time())
rng = np.random.default_rng(np.random.MT19937(seed=seed))

# regression objects
reg = Regression(max_degree, n, noise, rng, scale=False)

# mse[0, :] -> scaled
# mse[1, :] -> unscaled
mse_train = np.zeros((2, max_degree))
mse_test = np.zeros((2, max_degree))
r2_train = np.zeros((2, max_degree))
r2_test = np.zeros((2, max_degree))

features_5 = int((6) * (7) / 2)
betas_5 = np.zeros((2, features_5))
var_betas_5 = np.zeros((2, features_5))

for i, deg in enumerate(range(1, max_degree + 1)):
    print(f"degree: {deg}/{max_degree}", end="\r")
    
    ( mse_train[0, i], r2_train[0, i],
    mse_test[0, i], r2_test[0, i],
    betas_scaled, var_betas_scaled ) = reg.ordinary_least_squares(degree=deg, scale=True)
    
    ( mse_train[1, i], r2_train[1, i], 
    mse_test[1, i], r2_test[1, i],
    betas_unscaled, var_betas_unscaled ) = reg.ordinary_least_squares(degree=deg, scale=False)
    
    # save betas and var_betas
    if deg == 5:
        betas_5[0, :] = betas_scaled.reshape((betas_scaled.shape[0], ))
        betas_5[1, :] = betas_unscaled.reshape((betas_unscaled.shape[0], ))
        var_betas_5[0, :] = np.sqrt(var_betas_scaled.reshape((betas_scaled.shape[0], )))
        var_betas_5[1, :] = np.sqrt(var_betas_unscaled.reshape((betas_unscaled.shape[0], )))


# confidence interval beta values plots
plt.figure("Confidence intervals for beta values", figsize=(7, 9), dpi=80)

ax = plt.subplot(211)
plt.errorbar(np.arange(betas_5[0, :].shape[0]), betas_5[0], yerr=2*var_betas_5[0, :], fmt='xb', capsize=4)
plt.title("scaled")
plt.xlim((-1, betas_5[0].shape[0]+1))
plt.xlabel(r"$i$")
plt.ylabel(r"$\beta_i \pm 2\sigma$")

ax = plt.subplot(212)
plt.errorbar(np.arange(betas_5[1, :].shape[0]), betas_5[1, :], yerr=2*var_betas_5[1, :], fmt='xb', capsize=4)
plt.xlim((-1, betas_5[1, :].shape[0]+1))
plt.title("unscaled scaled")
plt.xlabel(r"$i$")
plt.ylabel(r"$\beta_i \pm 2\sigma$")

degrees = np.arange(1, max_degree + 1)

# plot MSE and R2 over complexity
plt.figure("MSE and R2 vs complexity", figsize=(11, 9), dpi=80)

# MSE scaled
plt.subplot(221)
plt.plot(degrees, mse_train[0, :], '-k', label="train")
plt.plot(degrees, mse_test[0, :], '--k', label="test")
plt.title("MSE scaled")
plt.xlabel(r"complexity")
plt.ylabel(r"MSE")
plt.legend()

# MSE unscaled
plt.subplot(222)
plt.plot(degrees, mse_train[1, :], '-k', label="train")
plt.plot(degrees, mse_test[1, :], '--k', label="test")
plt.title("MSE unscaled")
plt.xlabel(r"complexity")
plt.ylabel(r"MSE")
plt.legend()

# R2 scaled
plt.subplot(223)
plt.plot(degrees, r2_train[0, :], '-k', label="train")
plt.plot(degrees, r2_test[0, :], '--k', label="test")
plt.title("R2 scaled")
plt.xlabel(r"complexity")
plt.ylabel(r"R2")
plt.legend()

# R2 unscaled
plt.subplot(224)
plt.plot(degrees, r2_train[1, :], '-k', label="train")
plt.plot(degrees, r2_test[1, :], '--k', label="test")
plt.title("R2 unscaled")
plt.xlabel(r"complexity")
plt.ylabel(r"R2")
plt.legend()

plt.show()

