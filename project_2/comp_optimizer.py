import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter

from NeuralNetwork.optimizer.GradientDescent import GradientDescent
from NeuralNetwork.optimizer.StochasticGradientDescent import StochasticGradientDescent
from NeuralNetwork.optimizer.RMSprop import RMSprop
from NeuralNetwork.optimizer.NewtonMethod import NewtonMethod

from NeuralNetwork.cost_function.LinearRegression import LinearRegression
from NeuralNetwork.cost_function.LogisticRegression import LogisticRegression

from functions import *

from sklearn.model_selection import train_test_split

# params
n = 10000
deg = 5
noise = 0.1
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

X = create_X_2D(deg, x, y)

X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.25, random_state=seed)

theta_ols = ols(X_train, z_train)

# parameters for simulations
tol = 1e-7
iter_max = int(1e10)
# eta_vals = np.power(10.0, [-5, -4, -3, -2, -1])
eta_vals = np.array([0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1])
n_eta = eta_vals.shape[0]

# arrays
methods = ["GD", "SGD", "RMS", "newton"]
time = dict()
mse = dict()
epochs = dict()

for method in methods:
    time[method] = list()
    mse[method] = list()
    epochs[method] = list()

# Newton's method
lin_reg = LinearRegression(X_train, z_train, X_test, z_test)
optimizer_newton = NewtonMethod(lin_reg)
tmp = perf_counter()
newton_out = optimizer_newton.optimize(tol=tol, iter_max=iter_max, eta=0, random_state=seed, verbose=True)
time_newton = perf_counter() - tmp

time["newton"].append(time_newton)
mse["newton"].append(newton_out[2])
epochs["newton"].append(newton_out[1])

# Gradient methods
for i, eta in enumerate(eta_vals):

    print()
    print(f" -- ETA: {eta} --")
    print()

    lin_reg = LinearRegression(X_train, z_train, X_test, z_test)
    optimizer_GD = GradientDescent(lin_reg)
    tmp = perf_counter()    
    GD_out = optimizer_GD.optimize(tol=tol, iter_max=iter_max, eta=eta, random_state=seed, verbose=True)
    time_GD = perf_counter() - tmp
    
    time["GD"].append(time_GD)
    mse["GD"].append(GD_out[2])
    epochs["GD"].append(GD_out[1])

    lin_reg = LinearRegression(X_train, z_train, X_test, z_test)
    optimizer_SGD = StochasticGradientDescent(lin_reg, size_minibatches=5)
    tmp = perf_counter()
    SGD_out = optimizer_SGD.optimize(tol=tol, iter_max=iter_max, eta=eta, random_state=seed, verbose=True)
    time_SGD = perf_counter() - tmp
    
    time["SGD"].append(time_SGD)
    mse["SGD"].append(SGD_out[2])
    epochs["SGD"].append(SGD_out[1])

    lin_reg = LinearRegression(X_train, z_train, X_test, z_test)
    optimizer_RMS = RMSprop(lin_reg)
    tmp = perf_counter()
    RMS_out = optimizer_RMS.optimize(tol=tol, iter_max=iter_max, eta=eta, random_state=seed, verbose=True)
    time_RMS = perf_counter() - tmp
    
    time["RMS"].append(time_RMS)
    mse["RMS"].append(RMS_out[2])
    epochs["RMS"].append(RMS_out[1])


# write to file
for i in range(n_eta):
    with open(f"./results/comp_optimization/GD/GD_eta_{i}.txt", "w") as file:
        for epoch in range(1, epochs["GD"][i] + 2):
            mse_write = mse["GD"][i][epoch - 1]
            file.write(f"{epoch} {mse_write} \n")
                
for i in range(n_eta):
    with open(f"./results/comp_optimization/SGD/SGD_eta_{i}.txt", "w") as file:
        for epoch in range(1, epochs["SGD"][i] + 2):
            mse_write = mse["SGD"][i][epoch - 1]
            file.write(f"{epoch} {mse_write} \n")
                
for i in range(n_eta):
    with open(f"./results/comp_optimization/RMS/RMS_eta_{i}.txt", "w") as file:
        for epoch in range(1, epochs["RMS"][i] + 2):
            mse_write = mse["RMS"][i][epoch - 1]
            file.write(f"{epoch} {mse_write} \n")

with open(f"./results/comp_optimization/time.txt", "w") as file:
    for i, eta in enumerate(eta_vals):
        time_GD = time["GD"][i]
        time_SGD = time["SGD"][i]
        time_RMS = time["RMS"][i]
        file.write(f"{eta} {time_GD} {time_SGD} {time_RMS} \n")

with open(f"./results/comp_optimization/newton.txt", "w") as file:
    time_write = time["newton"][0]
    file.write(f"{time_write} \n")
    for epoch in range(1, epochs["newton"][0] + 2):
        mse_write = mse["newton"][0][epoch - 1]
        file.write(f"{epoch} {mse_write} \n")
