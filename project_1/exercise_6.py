import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import cross_val_score, KFold

from terrain import load_terrain, TERRAIN_1, TERRAIN_2
from functions import Regression, create_X_2D, scale_mean
from plots import plot_prediction_3D


# constants
max_degree_ols = 20
max_degree_cv_ols = 14 # Lower max degree for CV/bootstrap than ols calculations since it takes much longer
max_degree_cv = 22 # Higher max degree for ridge/lasso CV than OLS CV since that's where it gets interesting
seed = 0
max_bootstrap = 300
scissor = 1 # crop data to 100% of the full set (no cropping in final data)
downsample = 10 # downsample data to 10% of the cropped set
bootstrap_downsamples = [ 5, 4, 3, 2 ] # 20%, 25%, 33%, 50%
n_folds_vals = np.array([ 5, 7, 10 ]) # number of folds for CV
lambdas = np.logspace(-6, 1, 100) # lambda values to use for ridge/lasso regression
terrain_set = TERRAIN_1 # pick terrain file to open
noise = 1.0 # assumed constant used to compute the std

# Selectively turn on/off certain parts of the exercise
do_ols = False  #  essentially exercise 1 again
do_bootstrap_bv = False #               2
do_cv_bv = False #                      3
do_ridge_cv = False #                   4
do_lasso_cv = False #                   5


# Load data set
x, y, z = load_terrain(terrain_set, downsample=downsample, scissor=scissor, rng=None, plot=True, show_plot=False, save_fig=True)
print(x.shape[0], 'datapoints') # Show number of data points

# RNG
rng = np.random.default_rng(np.random.MT19937(seed=seed))

degrees_ols = np.arange(1, max_degree_ols + 1)
degrees_cv_ols = np.arange(1, max_degree_cv_ols + 1)
degrees_cv = np.arange(1, max_degree_cv + 1)



# -------------------------------
# Predictions for degrees 1..max_degree
# With and without scaling, split into train/test
# Computing confidence intervals for beta values for last degree
# -------------------------------

if do_ols:
    print("Computing MSEs/R2 for increasing degrees & beta confidence intervals...")

    # Regression object
    reg = Regression(max_degree_ols, x.shape[0], noise, rng, scale=False, data=(x, y, z))

    # mse[0, :] -> scaled
    # mse[1, :] -> unscaled
    mse_train = np.zeros((2, max_degree_ols))
    mse_test = np.zeros((2, max_degree_ols))
    r2_train = np.zeros((2, max_degree_ols))
    r2_test = np.zeros((2, max_degree_ols))

    # Keep track of betas/variance for degree 5 polynomial
    features_d5 = int((5 + 1) * (5 + 2) / 2)
    betas_d5 = np.zeros((2, features_d5))
    std_betas_d5 = np.zeros((2, features_d5))

    # Compute MSEs for OLS on all degrees from 1 to max_degree
    for i, deg in enumerate(range(1, max_degree_ols + 1)):
        print(f"degree: {deg}/{max_degree_ols}", end="\r")
        
        ( mse_train[0, i], r2_train[0, i],
        mse_test[0, i], r2_test[0, i],
        betas_scaled, var_betas_scaled ) = reg.ordinary_least_squares(degree=deg, scale=True)
        
        ( mse_train[1, i], r2_train[1, i], 
        mse_test[1, i], r2_test[1, i],
        betas_unscaled, var_betas_unscaled ) = reg.ordinary_least_squares(degree=deg, scale=False)
        
        # save betas and var_betas
        if deg == 5:
            betas_d5[0, :] = betas_scaled.reshape((betas_scaled.shape[0], ))
            betas_d5[1, :] = betas_unscaled.reshape((betas_unscaled.shape[0], ))
            std_betas_d5[0, :] = np.sqrt(var_betas_scaled.reshape((betas_scaled.shape[0], )))
            std_betas_d5[1, :] = np.sqrt(var_betas_unscaled.reshape((betas_unscaled.shape[0], )))


    # confidence interval beta values plots
    plt.figure("Confidence intervals for beta values", figsize=(7, 9), dpi=80)

    ax = plt.subplot(211)
    plt.errorbar(np.arange(betas_d5[0, :].shape[0]), betas_d5[0], yerr=2*std_betas_d5[0, :], fmt='xb', capsize=4)
    plt.title("scaled")
    plt.xlim((-1, betas_d5[0].shape[0]+1))
    plt.xlabel(r"$i$")
    plt.ylabel(r"$\beta_i \pm 2\sigma$")

    ax = plt.subplot(212)
    plt.errorbar(np.arange(betas_d5[1, :].shape[0]), betas_d5[1, :], yerr=2*std_betas_d5[1, :], fmt='xb', capsize=4)
    plt.xlim((-1, betas_d5[1, :].shape[0]+1))
    plt.title("unscaled scaled")
    plt.xlabel(r"$i$")
    plt.ylabel(r"$\beta_i \pm 2\sigma$")

    plt.savefig("./images/ex6_conf_int_beta.pdf", dpi=400)

    # plot MSE and R2 over complexity
    plt.figure("MSE and R2 vs complexity", figsize=(11, 9), dpi=80)

    # MSE scaled
    plt.subplot(221)
    plt.plot(degrees_ols, mse_train[0, :], '-k', label="train")
    plt.plot(degrees_ols, mse_test[0, :], '--k', label="test")
    plt.title("MSE scaled")
    plt.xlabel(r"complexity")
    plt.ylabel(r"MSE")
    plt.legend()

    # MSE unscaled
    plt.subplot(222)
    plt.plot(degrees_ols, mse_train[1, :], '-k', label="train")
    plt.plot(degrees_ols, mse_test[1, :], '--k', label="test")
    plt.title("MSE unscaled")
    plt.xlabel(r"complexity")
    plt.ylabel(r"MSE")
    plt.legend()

    # R2 scaled
    plt.subplot(223)
    plt.plot(degrees_ols, r2_train[0, :], '-k', label="train")
    plt.plot(degrees_ols, r2_test[0, :], '--k', label="test")
    plt.title("R2 scaled")
    plt.xlabel(r"complexity")
    plt.ylabel(r"R2")
    plt.legend()

    # R2 unscaled
    plt.subplot(224)
    plt.plot(degrees_ols, r2_train[1, :], '-k', label="train")
    plt.plot(degrees_ols, r2_test[1, :], '--k', label="test")
    plt.title("R2 unscaled")
    plt.xlabel(r"complexity")
    plt.ylabel(r"R2")
    plt.legend()

    plt.savefig("./images/ex6_mse_r2_comp.pdf", dpi=400)
    
    # Plot prediction to visually compare with original data
    plot_prediction_3D(betas_d5[0], 5, name=terrain_set + ' OLS prediction (degree ' + str(max_degree_ols) + ' polynomial)', show=False, save_fig=True)



# -------------------------------
# Bias-variance trade-off with bootstrap
# -------------------------------

if do_bootstrap_bv:

    print("Computing bias-variance trade-off with Bootstrap")

    plt.figure("bias-variance trade-off", figsize=(11, 9), dpi=80)

    # bootstrap for bias and var
    for j, ds in enumerate(bootstrap_downsamples):

        bx, by, bz = load_terrain(terrain_set, downsample=ds, scissor=scissor, rng=None, show_plot=False, plot=False, save_fig=False)
        n = bx.shape[0]

        # regression object
        b_reg = Regression(max_degree_cv_ols, bx.shape[0], noise, rng, data=(bx, by, bz))

        mse = np.zeros(max_degree_cv_ols)
        bias = np.zeros(max_degree_cv_ols)
        var = np.zeros(max_degree_cv_ols)
        
        for i, deg in enumerate(range(1, max_degree_cv_ols + 1)):
            mse[i], bias[i], var[i] = b_reg.bootstrap(degree=deg, max_bootstrap_cycle=max_bootstrap)

        plt.subplot(2, 2, j+1)
        
        plt.title(fr"$n={n}$")

        plt.plot(degrees_cv_ols, mse, '-r', label='MSE')
        plt.plot(degrees_cv_ols, var, '--k', label='var')
        plt.plot(degrees_cv_ols, bias, '--k', label='bias', alpha=0.40)
        plt.plot(degrees_cv_ols, bias + var, '-.k', label='var+bias')

        plt.xlabel(r"complexity")
        plt.legend()

    plt.savefig("./images/ex6_bv_bootstrap_various_n.pdf", dpi=400)

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
        reg = Regression(max_degree_cv_ols, x.shape[0], noise, rng, data=(x, y, z))

        mse_cv = np.zeros(max_degree_cv_ols)
        mse_cv_sk = np.zeros(max_degree_cv_ols)
        for i, deg in enumerate(degrees_cv_ols):

            # Compute own
            mse_cv[i] = reg.k_folds_cross_validation(degree=deg, n_folds=n_folds)

            # Compute SKLearn
            sk_kfold = KFold(n_splits=n_folds, shuffle=True)
            mse_cv_sk[i] = np.mean(-cross_val_score(LinearRegression(fit_intercept=False), create_X_2D(deg, x, y), z, cv=sk_kfold, scoring="neg_mean_squared_error"))
        
        plt.subplot(2, 2, j+1)
        
        plt.plot(degrees_cv_ols, mse_cv, '-k', label="Our code")
        plt.plot(degrees_cv_ols, mse_cv_sk, 'b--', label="Sklearn") # Plot against sklearn's
        plt.xlabel(r"complexity")
        plt.ylabel(r"MSE")
        plt.legend(loc="best")
        plt.title(f"k-folds cross validation with k={n_folds}")

    # compare with bootstrap
    reg = Regression(max_degree_cv_ols, x.shape[0], noise, rng, data=(x, y, z))
    mse = np.zeros(max_degree_cv_ols)
    bias = np.zeros(max_degree_cv_ols)
    var = np.zeros(max_degree_cv_ols)
    for i, deg in enumerate(range(1, max_degree_cv_ols + 1)):
        mse[i], bias[i], var[i] = reg.bootstrap(degree=deg, max_bootstrap_cycle=max_bootstrap)

    plt.subplot(2, 2, 4)
    plt.plot(degrees_cv_ols, mse, '-r')
    plt.xlabel(r"complexity")
    plt.ylabel(r"MSE")
    plt.title(f"bootstrap with n_cycles={max_bootstrap}")

    plt.savefig("./images/ex6_bv_bootstrap_cv_comp.pdf", dpi=400)

# -------------------------------
# CV with ridge
# -------------------------------

if do_ridge_cv:

    reg = Regression(max_degree_cv, x.shape[0], noise, rng, data=(x, y, z), with_std=False)

    n_lambdas = lambdas.shape[0]

    # min lmd and deg arrays
    min_mse = np.zeros(n_folds_vals.shape[0] + 1)
    lmd_min = np.zeros(n_folds_vals.shape[0] + 1)
    deg_min = np.zeros(n_folds_vals.shape[0] + 1)

    # bootstrap for MSE
    mse = np.zeros((n_lambdas, max_degree_cv))

    for j, deg in enumerate(degrees_cv):
        for i, lmd in enumerate(lambdas):
            mse[i, j], _, _ = reg.bootstrap(degree=deg, max_bootstrap_cycle=max_bootstrap, lmd=lmd)

    min_mse_where = np.where(mse == np.min(mse))
    lmd_min[0] = lambdas[min_mse_where[0][0]]
    deg_min[0] = degrees_cv[min_mse_where[1][0]]
    min_mse[0] = mse[min_mse_where[0][0], min_mse_where[1][0]]

    # mse vs (lambdas, degs) for bootstrap
    plt.figure(f"bootstrap Ridge", figsize=(11, 9), dpi=80)

    plt.contourf(np.log10(lambdas), degrees_cv, mse.T)
    plt.plot(np.log10(lambdas[min_mse_where[0][0]]), degrees_cv[min_mse_where[1][0]], 'or')
    plt.ylabel("degrees",fontsize=14)
    plt.xlabel("lambdas",fontsize=14)
    plt.colorbar()

    plt.savefig(f"./images/ex6_bootstrap_bsc_{max_bootstrap}_ridge.pdf", dpi=400)

    for idx, n_folds in enumerate(n_folds_vals):
        # cross validation for MSE
        mse = np.zeros((n_lambdas, max_degree_cv))

        for j, deg in enumerate(degrees_cv):
            for i, lmd in enumerate(lambdas):
                mse[i, j] = reg.k_folds_cross_validation(degree=deg, n_folds=n_folds, lmd=lmd)

        min_mse_where = np.where(mse == np.min(mse))
        lmd_min[idx + 1] = lambdas[min_mse_where[0][0]]
        deg_min[idx + 1] = degrees_cv[min_mse_where[1][0]]
        min_mse[idx + 1] = mse[min_mse_where[0][0], min_mse_where[1][0]]

        # mse vs (lambdas, degs) for cross validation
        plt.figure(fr"cross validation Ridge, {n_folds} folds", figsize=(11, 9), dpi=80)

        plt.contourf(np.log10(lambdas), degrees_cv, mse.T)
        plt.plot(np.log10(lambdas[min_mse_where[0][0]]), degrees_cv[min_mse_where[1][0]], 'or')
        plt.ylabel("degrees",fontsize=14)
        plt.xlabel("lambdas",fontsize=14)
        plt.colorbar()

        plt.savefig(f"./images/ex6_cv_n_folds_{n_folds}_lasso.pdf", dpi=400)
    
    # save min to file
    with open("./ex6_min_ridge.txt", "w") as file:
        file.write("Bootstrap: \n")
        file.write(f"mse: {min_mse[0]}; lmd: {lmd_min[0]}; deg: {deg_min[0]} \n")
        
        file.write("Cross Validation: \n")
        for i, n_folds in enumerate(n_folds_vals):    
            file.write(f"n_folds: {n_folds}; mse: {min_mse[i + 1]}; lmd: {lmd_min[i + 1]}; deg: {deg_min[i + 1]} \n")
    
# -------------------------------
# CV with lasso
# -------------------------------

if do_lasso_cv:

    reg = Regression(max_degree_cv, x.shape[0], noise, rng, data=(x, y, z), with_std=False)

    n_lambdas = lambdas.shape[0]
    
    # min lmd and deg arrays
    min_mse_where = np.where(mse == np.min(mse))
    lmd_min[0] = lambdas[min_mse_where[0][0]]
    deg_min[0] = degrees_cv[min_mse_where[1][0]]
    min_mse[0] = mse[min_mse_where[0][0], min_mse_where[1][0]]

    # bootstrap for MSE
    mse = np.zeros((n_lambdas, max_degree_cv))

    for j, deg in enumerate(degrees_cv):
        for i, alpha in enumerate(lambdas):
            mse[i, j], _, _ = reg.bootstrap(degree=deg, max_bootstrap_cycle=max_bootstrap, alpha=alpha)

    min_mse_where = np.where(mse == np.min(mse))
    lmd_min[0] = lambdas[min_mse_where[0][0]]
    deg_min[0] = degrees_cv[min_mse_where[1][0]]
    min_mse[0] = mse[min_mse_where[0][0], min_mse_where[1][0]]

    # mse vs (lambdas, degs) for bootstrap
    plt.figure(f"bootstrap Lasso", figsize=(11, 9), dpi=80)
    plt.contourf(np.log10(lambdas), degrees_cv, mse.T)
    plt.plot(np.log10(lambdas[min_mse_where[0][0]]), degrees_cv[min_mse_where[1][0]], 'or')
    plt.ylabel("degrees",fontsize=14)
    plt.xlabel("lambdas",fontsize=14)
    plt.colorbar()

    plt.savefig("./images/ex6_bootstrap_bsc_{max_bootstrap}_lasso.pdf", dpi=400)

    for idx, n_folds in enumerate(n_folds_vals):
        # cross validation for MSE
        mse = np.zeros((n_lambdas, max_degree_cv))

        for j, deg in enumerate(degrees_cv):
            for i, alpha in enumerate(lambdas):
                mse[i, j] = reg.k_folds_cross_validation(degree=deg, n_folds=n_folds, alpha=alpha)

        min_mse_where = np.where(mse == np.min(mse))
        lmd_min[idx + 1] = lambdas[min_mse_where[0][0]]
        deg_min[idx + 1] = degrees_cv[min_mse_where[1][0]]
        min_mse[idx + 1] = mse[min_mse_where[0][0], min_mse_where[1][0]]
        
        # mse vs (lambdas, degs) for cross validation
        plt.figure(fr"cross validation Lasso, {n_folds} folds", figsize=(11, 9), dpi=80)

        plt.contourf(np.log10(lambdas), degrees_cv, mse.T)
        plt.plot(np.log10(lambdas[min_mse_where[0][0]]), degrees_cv[min_mse_where[1][0]], 'or')
        plt.ylabel("degrees",fontsize=14)
        plt.xlabel("lambdas",fontsize=14)
        plt.colorbar()

        plt.savefig(f"./images/ex6_cv_n_folds_{n_folds}_lasso.pdf", dpi=400)
    
    # save min to file
    with open("./ex6_min_lasso.txt", "w") as file:
        file.write("Bootstrap: \n")
        file.write(f"mse: {min_mse[0]}; lmd: {lmd_min[0]}; deg: {deg_min[0]} \n")
        
        file.write("Cross Validation: \n")
        for i, n_folds in enumerate(n_folds_vals):    
            file.write(f"n_folds: {n_folds}; mse: {min_mse[i + 1]}; lmd: {lmd_min[i + 1]}; deg: {deg_min[i + 1]} \n")

