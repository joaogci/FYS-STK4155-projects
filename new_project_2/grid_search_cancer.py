import numpy as np
from Network import NeuralNetwork

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from Activation import *
from Cost import *

import seaborn as sns
import matplotlib.pyplot as plt

seed = 1337

sns.set_theme(style="white")

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

# params
epochs = 1000
size_batches = 5
eta_vals = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
reg_vals = [0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]

accuracy = np.zeros((len(eta_vals), len(reg_vals)))

for eta_i, eta in enumerate(eta_vals):
    for reg_i, regularization in enumerate(reg_vals):
        print()
        print(f"eta: {eta_i + 1}/{len(eta_vals)}; reg: {reg_i + 1}/{len(reg_vals)}")
        
        neural_network = NeuralNetwork(30, random_state=seed)
        neural_network.add_layer(20, sigmoid())
        neural_network.add_layer(20, sigmoid())
        neural_network.add_layer(1, sigmoid())

        learning_rate = lambda x: eta
        neural_network.train(X_train, y_train, grad_cross_entropy, epochs, learning_rate, size_batches, regularization)

        pred = neural_network.predict(X_test)
        pred = pred.round()
        accuracy[eta_i, reg_i] = np.sum(pred == y_test)/y_test.shape[1]

sns.set()

fig, ax = plt.subplots(figsize = (10, 10))
sns.heatmap(accuracy, annot=True, ax=ax, cmap="viridis")
ax.set_title("Training Accuracy")
ax.set_ylabel("$\eta$")
ax.set_xlabel("$\lambda$")
plt.show()

