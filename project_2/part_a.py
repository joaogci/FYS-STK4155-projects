import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor

from NeuralNetwork.optimizer.StochasticGradientDescent import StochasticGradientDescent
from NeuralNetwork.optimizer.GradientDescent import GradientDescent
from NeuralNetwork.optimizer.NewtonMethod import NewtonMethod
from NeuralNetwork.cost_function.LinearRegression import LinearRegression
from functions import *

def main():
    
    # parameters
    n = 1000
    noise = 0.25
    seed = 1337

    # rng 
    rng = np.random.default_rng(np.random.MT19937(seed=seed))

    # data
    x = np.sort(rng.uniform(0, 1, int(np.sqrt(n))))
    y = np.sort(rng.uniform(0, 1, int(np.sqrt(n))))

    x, y = np.meshgrid(x, y)

    z = franke_function(x, y)
    z += noise * rng.normal(0, 1, z.shape)

    x = np.ravel(x)
    y = np.ravel(y)
    z = np.ravel(z)

    X = create_X_2D(5, x, y)

    X_train, X_test, y_train, y_test = train_test_split(X, z, test_size=0.25, random_state=seed)

    # 1) -> plot MSE as number of epochs for GD and SGD
    #       with one bad and good case for each method
    #       all of this for OLS
    part_1(X_train, X_test, y_train, y_test, seed, epochs=200, 
           eta_1=0.01, eta_2=0.001, size_batches=5)

    # 2) -> plot of MSE vs size of batches with constant eta
    part_2(X_train, X_test, y_train, y_test, seed, epochs=200,
           eta=0.001, batch_vals=[1, 5, 10, 50, 100])
    
    # 3) -> for SGD and OLS perform grid search for eta and size of batches
    for epochs in [100, 500, 1000, 5000, 10000]:
        part_3(X_train, X_test, y_train, y_test, seed, epochs=epochs, 
            eta_vals=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5], 
            batch_vals=[1, 5, 10, 50, 100])
    
    # 4) -> for SGD and Ridge perform grid search for eta and lambda
    #       size of batches is 5
    for epochs in [100, 500, 1000, 5000, 10000]:
        part_4(X_train, X_test, y_train, y_test, seed, epochs=epochs, 
            eta_vals=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5], batch_size=5, 
            reg_vals=[0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5])


def part_1(X_train, X_test, y_train, y_test, seed, epochs, eta_1, eta_2, batch_size):

    lin_reg = LinearRegression(X_train, y_train, X_test, y_test)
    optimizer_SGD = StochasticGradientDescent(lin_reg, size_minibatches=batch_size)
    out_SGD_1 = optimizer_SGD.optimize(iter_max=epochs, eta=eta_1, random_state=seed, tol=0, verbose=True)
    optimizer_SGD = StochasticGradientDescent(lin_reg, size_minibatches=batch_size)
    out_SGD_2 = optimizer_SGD.optimize(iter_max=epochs, eta=eta_2, random_state=seed, tol=0, verbose=True)

    lin_reg = LinearRegression(X_train, y_train, X_test, y_test)
    optimizer_GD = GradientDescent(lin_reg)
    out_GD_1 = optimizer_GD.optimize(iter_max=epochs, eta=eta_1, random_state=seed, tol=0, verbose=True)
    optimizer_GD = GradientDescent(lin_reg)
    out_GD_2 = optimizer_GD.optimize(iter_max=epochs, eta=eta_2, random_state=seed, tol=0, verbose=True)

    plt.figure("MSE vs epochs")

    plt.plot(np.arange(1, out_GD_1[1] + 2), out_GD_1[2], '--r', label=f'GD with eta={eta_1}')
    plt.plot(np.arange(1, out_GD_2[1] + 2), out_GD_2[2], '-r', label=f'GD with eta={eta_2}')

    plt.plot(np.arange(1, out_SGD_1[1] + 2), out_SGD_1[2], '--b', label=f'SGD with eta={eta_1}')
    plt.plot(np.arange(1, out_SGD_2[1] + 2), out_SGD_2[2], '-b', label=f'SGD with eta={eta_2}')

    plt.xlabel("epochs")
    plt.ylabel("MSE")
    plt.title("MSE vs epochs for GD and SGD with different learning rates")
    plt.legend()
    
    plt.savefig(f"./figs/part_a/1_mse_eta_epochs_{epochs}_s_bathes_{batch_size}.pdf", dpi=400)

def part_2(X_train, X_test, y_train, y_test, seed, epochs, eta, batch_vals):
    
    plt.figure("MSE vs batch size")
    
    for size_batches in batch_vals:
        lin_reg = LinearRegression(X_train, y_train, X_test, y_test)
        optimizer_SGD = StochasticGradientDescent(lin_reg, size_minibatches=size_batches)
        out_SGD = optimizer_SGD.optimize(iter_max=epochs, eta=eta, random_state=seed, tol=0, verbose=True)
        plt.plot(np.arange(1, out_SGD[1] + 2), out_SGD[2], label=f'batch_size={size_batches}')
    
    lin_reg = LinearRegression(X_train, y_train, X_test, y_test)
    optimizer_GD = GradientDescent(lin_reg)
    out_GD = optimizer_GD.optimize(iter_max=epochs, eta=eta, random_state=seed, tol=0, verbose=True)
    plt.plot(np.arange(1, out_GD[1] + 2), out_GD[2], label=f'full batch (GD)')
    
    plt.xlabel("epochs")
    plt.ylabel("MSE")
    plt.title("MSE vs epochs for SGD with different batch size")
    plt.legend()
    
    plt.savefig(f"./figs/part_a/2_mse_batch_size_epochs_{epochs}_eta_{eta}.pdf", dpi=400)

def part_3(X_train, X_test, y_train, y_test, seed, epochs, eta_vals, batch_vals):

    mse = np.zeros((len(eta_vals), len(batch_vals)))
    
    for eta_i, eta in enumerate(eta_vals):
        for batch_i, batch_size in enumerate(batch_vals):
            lin_reg = LinearRegression(X_train, y_train, X_test, y_test)
            optimizer_SGD = StochasticGradientDescent(lin_reg, size_minibatches=batch_size)
            out_SGD = optimizer_SGD.optimize(iter_max=epochs, eta=eta, random_state=seed, tol=0, verbose=True)
            
            mse[eta_i, batch_i] = mean_squared_error(y_test, X_test @ out_SGD[0])
     
    sns.set()
    fig, ax = plt.subplots()
    sns.heatmap(mse, annot=True, ax=ax, cmap="viridis")
    ax.set_title(f"Training MSE for SGD with {epochs} epochs")
    ax.set_ylabel("$\eta$")
    ax.set_xlabel("batch size")
    
    plt.savefig(f"./figs/part_a/3_mse_eta_size_batch_epochs_{epochs}.pdf", dpi=400)
    
def part_4(X_train, X_test, y_train, y_test, seed, epochs, eta_vals, batch_size, reg_vals):
    mse = np.zeros((len(eta_vals), len(reg_vals)))
    
    for eta_i, eta in enumerate(eta_vals):
        for reg_i, reg in enumerate(reg_vals):
            lin_reg = LinearRegression(X_train, y_train, X_test, y_test, regularization=reg)
            optimizer_SGD = StochasticGradientDescent(lin_reg, size_minibatches=batch_size)
            out_SGD = optimizer_SGD.optimize(iter_max=epochs, eta=eta, random_state=seed, tol=0, verbose=True)
            
            mse[eta_i, reg_i] = mean_squared_error(y_test, X_test @ out_SGD[0])
     
    sns.set()
    fig, ax = plt.subplots()
    sns.heatmap(mse, annot=True, ax=ax, cmap="viridis")
    ax.set_title(f"Training MSE for SGD with {epochs} epochs")
    ax.set_ylabel("$\eta$")
    ax.set_xlabel("$\lambda$")
    
    plt.savefig(f"./figs/part_a/4_mse_eta_reg_epochs_{epochs}.pdf", dpi=400)

if __name__ == "__main__":
    main()
