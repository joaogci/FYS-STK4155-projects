import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from NeuralNetwork.cost_function.LogisticRegression import LogisticRegression
from NeuralNetwork.optimizer.StochasticGradientDescent import StochasticGradientDescent
from NeuralNetwork.activation_function.Sigmoid import Sigmoid

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

seed = 1337

cancerdata = load_breast_cancer() #(569, 30)

data = cancerdata['data']
target = cancerdata['target']

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.25, random_state=seed)

X_train_s = X_train - np.mean(X_train, axis=0, keepdims=True)
X_test_s = X_test - np.mean(X_train, axis=0, keepdims=True)

sigmoid = lambda z: 1 / (1 + np.exp(- z))

# params
epochs = 500
size_batches = 5
eta_vals = [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
reg_vals = [1e1, 1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 0]

accuracy = np.zeros((len(eta_vals), len(reg_vals)))

for eta_i, eta in enumerate(eta_vals):
    for reg_i, regularization in enumerate(reg_vals):
        print()
        print(f"eta: {eta_i + 1}/{len(eta_vals)}; reg: {reg_i + 1}/{len(reg_vals)}")
        
        log_reg = LogisticRegression(X_train_s, y_train, X_test_s, y_test)
        optimizer = StochasticGradientDescent(log_reg, size_minibatches=5)
        theta_SGD = optimizer.optimize(iter_max=epochs, eta=eta, regularization=regularization, random_state=seed, tol=-1)

        pred = sigmoid(X_test_s.dot(theta_SGD)).round()
        accuracy[eta_i, reg_i] = np.sum(pred == y_test)/len(y_test)

sns.set()

fig, ax = plt.subplots()
sns.heatmap(accuracy, annot=True, ax=ax, cmap="viridis", cbar_kws={'label': 'Accuracy'})
ax.set_title(f"Test Accuracy for SGD with {epochs} epochs")
ax.set_ylabel("$\eta$")
ax.set_xlabel("$\lambda$")
ax.set_yticklabels(eta_vals)
ax.set_xticklabels(reg_vals)

plt.savefig(f"./figs/part_e/1_cancer_log_reg_{epochs}_epochs.pdf", dpi=400)


