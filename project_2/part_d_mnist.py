import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from NeuralNetwork.Model import Model
from NeuralNetwork.Layer import HiddenLayer, OutputLayer
from NeuralNetwork.ActivationFunctions import Sigmoid, ReLU, ELU, LeakyReLU, Linear, Tanh, Softmax
from NeuralNetwork.cost_function.LinearRegression import LinearRegression

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler

seed = 1337
# Extract data for ScikitLearn
# Extract data for ScikitLearn
digits = load_digits()
data = digits.data
target = digits.target

X_train, X_test, y_train_wrong_dim, y_test = train_test_split(data, target, test_size=0.25, random_state=seed)

y_train = np.zeros((y_train_wrong_dim.shape[0],10))
for i in range(len(y_train_wrong_dim)):
    y_train[ i, y_train_wrong_dim[i] ] = 1

y_train = y_train.T

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

fig, ax = plt.subplots()
sns.heatmap(accuracy, annot=True, ax=ax, cmap="viridis", cbar_kws={'label': 'Accuracy'})
ax.set_title("Test Accuracy")
ax.set_ylabel("$\eta$")
ax.set_xlabel("$\lambda$")
ax.set_xticklabels(reg_vals)
ax.set_yticklabels(eta_vals)
plt.savefig("./figs/part_d/1_grid_search_mnist.pdf", dpi=400)


