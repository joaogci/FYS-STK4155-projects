import numpy as np
from Network import NeuralNetwork

from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

seed = 1337

# Activation function(s) and it's derivatives
sigmoid = lambda x: 1 / (1 + np.exp(- x))
grad_sigmoid = lambda x: np.multiply(np.exp(- x), np.power((1 + np.exp(- x)), -2))

# Cost functions and it's derivatives
# Mean squared error
mean_squared_error = lambda target, pred: (pred - target)**2 / target.shape[0]
grad_MSE = lambda target, pred: 2 * (pred - target) / target.shape[0]
# Cross-entropy (i.e. log-likelihood)
cross_entroy = lambda target, pred: - (target - pred) / target.shape[0]
grad_cross_ent = lambda target, pred: - (target - pred) / target.shape[0]

"""
Test if we can determine the type of tumor for breast cancer data.
"""
# Extract data for ScikitLearn
cancer = load_breast_cancer()
data = cancer.data
target = cancer.target

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.25, random_state=seed)

y_train = y_train.reshape(1, -1)
y_test = y_test.reshape(1, -1)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train).T
X_test = scaler.transform(X_test).T

neural_network = NeuralNetwork(30, random_state=seed)
neural_network.add_layer(20, (sigmoid, grad_sigmoid))
neural_network.add_layer(20, (sigmoid, grad_sigmoid))
neural_network.add_layer(1, (sigmoid, grad_sigmoid))

epochs = 1000
size_batches = 5
learning_rate = lambda x: 0.1
neural_network.train(X_train, y_train, grad_cross_ent, epochs, learning_rate, size_batches)

pred = neural_network.predict(X_test)
pred = pred.round()

print(f"accuracy: {np.sum(pred == y_test)}/{y_test.shape[1]} = {np.sum(pred == y_test)/y_test.shape[1]:.3f}")
