
from numpy.core.numeric import cross
from terrain import load_terrain, TERRAIN_1, TERRAIN_2
from functions import Regression, create_X_2D
from plots import plot_prediction_3D
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
import numpy as np
import matplotlib.pyplot as plt



# constants
max_degree = 20 #50
seed = 0
max_bootstrap = 15 # 100
scissor = 0.05 # crop data to 5% of the full set
downsample = 4 #2 # downsample data to 50% of the cropped set
bootstrap_downsamples = [ 5, 4, 3, 2 ] # 20%, 25%, 33%, 50%
n_folds_vals = [ 5, 7, 10 ] # number of folds for CV
terrain_set = TERRAIN_1 # pick terrain file to open
noise = 1.0 # assumed constant used to compute the std

# Selectively turn on/off certain parts of the exercise
do_ols = False
do_bootstrap_bv = False
do_cv_bv = True


# Load data set
# TODO: enable random sampling with rng=True to see the difference
x, y, z = load_terrain(terrain_set, downsample=downsample, scissor=scissor, rng=None, plot=True, show_plot=False)
print(x.shape[0], 'datapoints') # Show number of data points

# RNG
rng = np.random.default_rng(np.random.MT19937(seed=seed))

degrees = np.arange(1, max_degree + 1)



# -------------------------------
# Predictions for degrees 1..max_degree
# With and without scaling, split into train/test
# Computing confidence intervals for beta values for last degree
# -------------------------------

if do_ols:
    print("Computing MSEs/R2 for increasing degrees & beta confidence intervals...")

    # Regression object
    reg = Regression(max_degree, x.shape[0], noise, rng, scale=False, data=(x, y, z))

    # mse[0, :] -> scaled
    # mse[1, :] -> unscaled
    mse_train = np.zeros((2, max_degree))
    mse_test = np.zeros((2, max_degree))
    r2_train = np.zeros((2, max_degree))
    r2_test = np.zeros((2, max_degree))

    # Keep track of final betas/variance
    features_last = int((max_degree + 1) * (max_degree + 2) / 2)
    betas_last = np.zeros((2, features_last))
    std_betas_last = np.zeros((2, features_last))

    # Compute MSEs for OLS on all degrees from 1 to max_degree
    for i, deg in enumerate(range(1, max_degree + 1)):
        print(f"degree: {deg}/{max_degree}", end="\r")
        
        ( mse_train[0, i], r2_train[0, i],
        mse_test[0, i], r2_test[0, i],
        betas_scaled, var_betas_scaled ) = reg.ordinary_least_squares(degree=deg, scale=True)
        
        ( mse_train[1, i], r2_train[1, i], 
        mse_test[1, i], r2_test[1, i],
        betas_unscaled, var_betas_unscaled ) = reg.ordinary_least_squares(degree=deg, scale=False)
        
        # save betas and var_betas
        if deg == max_degree:
            betas_last[0, :] = betas_scaled.reshape((betas_scaled.shape[0], ))
            betas_last[1, :] = betas_unscaled.reshape((betas_unscaled.shape[0], ))
            std_betas_last[0, :] = np.sqrt(var_betas_scaled.reshape((betas_scaled.shape[0], )))
            std_betas_last[1, :] = np.sqrt(var_betas_unscaled.reshape((betas_unscaled.shape[0], )))


    # confidence interval beta values plots
    plt.figure("Confidence intervals for beta values", figsize=(7, 9), dpi=80)

    ax = plt.subplot(211)
    plt.errorbar(np.arange(betas_last[0, :].shape[0]), betas_last[0], yerr=2*std_betas_last[0, :], fmt='xb', capsize=4)
    plt.title("scaled")
    plt.xlim((-1, betas_last[0].shape[0]+1))
    plt.xlabel(r"$i$")
    plt.ylabel(r"$\beta_i \pm 2\sigma$")

    ax = plt.subplot(212)
    plt.errorbar(np.arange(betas_last[1, :].shape[0]), betas_last[1, :], yerr=2*std_betas_last[1, :], fmt='xb', capsize=4)
    plt.xlim((-1, betas_last[1, :].shape[0]+1))
    plt.title("unscaled scaled")
    plt.xlabel(r"$i$")
    plt.ylabel(r"$\beta_i \pm 2\sigma$")

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

    # Plot prediction to visually compare with original data
    plot_prediction_3D(betas_last[0], max_degree, name=terrain_set + ' OLS prediction (degree ' + str(max_degree) + ' polynomial)', show=False)


# -------------------------------
# Bias-variance trade-off with bootstrap
# -------------------------------

if do_bootstrap_bv:

    print("Computing bias-variance trade-off with Bootstrap")

    plt.figure("bias-variance trade-off", figsize=(11, 9), dpi=80)

    # bootstrap for bias and var
    for j, ds in enumerate(bootstrap_downsamples):

        bx, by, bz = load_terrain(terrain_set, downsample=ds, scissor=scissor, rng=None)
        n = bx.shape[0]

        # regression object
        b_reg = Regression(max_degree, bx.shape[0], noise, rng, data=(bx, by, bz))

        mse = np.zeros(max_degree)
        bias = np.zeros(max_degree)
        var = np.zeros(max_degree)
        
        for i, deg in enumerate(range(1, max_degree + 1)):
            mse[i], bias[i], var[i] = b_reg.bootstrap(degree=deg, max_bootstrap_cycle=max_bootstrap)

        plt.subplot(2, 2, j+1)
        
        plt.title(fr"$n={n}$")

        plt.plot(degrees, mse, '-r', label='MSE')
        plt.plot(degrees, var, '--k', label='var')
        plt.plot(degrees, bias, '--k', label='bias', alpha=0.40)
        plt.plot(degrees, bias + var, '-.k', label='var+bias')

        plt.xlabel(r"complexity")
        plt.legend()



# -------------------------------
# Bias-variance trade-off with CV
# -------------------------------

if do_cv_bv:
    
    print("Computing bias-variance trade-off with CV")

    # figure plot
    plt.figure("MSE comparison", figsize=(11, 9), dpi=80)

    # cross validation
    for j, n_folds in enumerate(n_folds_vals):
        # regression object
        reg = Regression(max_degree, x.shape[0], noise, rng, data=(x, y, z))

        mse_cv = np.zeros(max_degree)
        for i, deg in enumerate(degrees):
            ...
            model = Ridge()
            X = create_X_2D(deg, x, y)
            mse_cv[i] = np.mean(-cross_val_score(model, X, z, cv=n_folds, scoring="neg_mean_squared_error"))
            # mse_cv[i] = reg.k_folds_cross_validation(degree=deg, n_folds=n_folds)
        
        plt.subplot(2, 2, j+1)
        
        plt.plot(degrees, mse_cv, '-k')
        plt.xlabel(r"complexity")
        plt.ylabel(r"MSE")
        plt.title(f"k-folds cross validation with k={n_folds}")

    # compare with bootstrap (re-use results from previous section)
    reg = Regression(max_degree, x.shape[0], noise, rng, data=(x, y, z))
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