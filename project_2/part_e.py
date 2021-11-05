import numpy as np
import matplotlib.pyplot as plt

from NeuralNetwork.cost_function.LogisticRegression import LogisticRegression
from NeuralNetwork.optimizer.StochasticGradientDescent import StochasticGradientDescent
from NeuralNetwork.activation_function.Sigmoid import Sigmoid

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

seed = 1
rng = np.random.default_rng(np.random.MT19937(seed=seed))

cancerdata = load_breast_cancer() #(569, 30)

data = cancerdata['data']
target = cancerdata['target']

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=seed)

X_train_s = X_train - np.mean(X_train, axis=0, keepdims=True)
X_test_s = X_test - np.mean(X_train, axis=0, keepdims=True)

sigmoid = lambda z: 1 / (1 + np.exp(- z))

log_reg = LogisticRegression(X_train_s, y_train, X_test_s, y_test)
optimizer = StochasticGradientDescent(log_reg, size_minibatches=5, rng=rng)
theta_SGD = optimizer.optimize(iter_max=int(1e4), eta=0.000025)

pred = sigmoid(X_test_s.dot(theta_SGD)).round()
print(np.sum(pred == y_test),"/", len(y_test))
print(f"accuracy: {np.sum(pred == y_test)/len(y_test)}")


