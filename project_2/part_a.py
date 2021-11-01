import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor

from time import time

from NeuralNetwork.StochasticGradientDescent import StochasticGradientDescent
from NeuralNetwork.cost_function.LinearRegression import  LinearRegression


# parameters
n = 1000
M = 5 # size of each minibatch 
m = n // M # number of minibatches
n_epochs = 100
t0 = 40
t1 = 50


# random number generator
seed = 10
rng = np.random.default_rng(np.random.MT19937(seed=seed))

# compute the function
x = rng.uniform(0, 1, size=n)
X = np.c_[np.ones((n, 1)), x, x**2]
y = 4 + 3 * x**2 + rng.normal(0, 1, size=n)

linear_reg = LinearRegression(X, y)
sgd = StochasticGradientDescent(linear_reg, M, t0, t1, rng)
theta_sgd_class = sgd.optimize()
theta_sgd_class_autograd = sgd.optimize_autograd()


# learning rate and grad of MSE
eta = lambda t: t0 / (t + t1)
grad_MSE = lambda xi, yi, theta: 2 * xi.T @ ((xi @ theta) - yi)
theta = np.zeros(X.shape[1])

# SGD
for epoch in range(1, n_epochs + 1):
    for i in range(m):
        # pick the kth minibath 
        k = rng.integers(m)
        theta = theta - eta(epoch * m + i) * grad_MSE(X[k*M:(k+1)*M], y[k*M:(k+1)*M], theta)

print("theta from class: ")
print(theta_sgd_class)


print("theta from class autograd: ")
print(theta_sgd_class_autograd)

print("theta from SGD: ")
print(theta)

print("theta from OLS: ")
print(np.linalg.pinv(X.T @ X) @ X.T @ y)
        
sgdreg = SGDRegressor(max_iter=n_epochs, penalty=None, eta0=t0/t1, random_state=seed)
sgdreg.fit(X, y)
print("theta from sklearn SGD: ")
print(sgdreg.coef_)




