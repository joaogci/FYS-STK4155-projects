import numpy as np
from Network import NeuralNetwork

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

seed = 1337

sigmoid = lambda x: 1 / (1 + np.exp(- x))
grad_sigmoid = lambda x: np.multiply(np.exp(- x), np.power((1 + np.exp(- x)), -2))

grad_MSE = lambda target, pred: 2 * (pred - target) / target.shape[0]
grad_cross_ent = lambda target, pred: - (target - pred) / target.shape[0]

# inputs = np.matrix([
#     [0, 0, 1, 1],
#     [0, 1, 0, 1]
# ])
# targets = np.matrix([ 
#     [0, 1, 1, 0],   # xor
#     [0, 1, 1, 1],   # or
#     [0, 0, 0, 1]    # and
# ])

# neural_network = NeuralNetwork(2, random_state=seed)
# neural_network.add_layer(16, (sigmoid, grad_sigmoid))
# neural_network.add_layer(3, (sigmoid, grad_sigmoid))

# output = neural_network.predict(inputs)
# print(output)

# for i in range(1, 1000+1):
#     print(f"{i}/1000")
#     neural_network.back_propagation(inputs, targets, grad_cross_ent)

# output = neural_network.predict(inputs)
# print(output)
# print(output.round())


# print(f"number of layers: {neural_network.n_layers}")
# for layer in neural_network.layers:
#     print()
#     print(f"shape of weights matrix: {layer.weights.shape}")
#     print(f"shape of bias matrix: {layer.biases.shape}")
# print("done!")

# breast cancer
cancer = load_breast_cancer()
data = cancer.data
target = cancer.target

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.25, random_state=seed)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train).T
X_test = scaler.transform(X_test).T

neural_network = NeuralNetwork(30, random_state=seed)
neural_network.add_layer(20, (sigmoid, grad_sigmoid))
neural_network.add_layer(20, (sigmoid, grad_sigmoid))
neural_network.add_layer(1, (sigmoid, grad_sigmoid))

for i in range(1, 10000+1):
    print(f"{i}/10000")
    neural_network.back_propagation(X_train, y_train, grad_cross_ent)

# print(X_train.shape)
# print(X_train.shape)

pred = neural_network.predict(X_test)
pred = pred.round()

print(f"accuracy: {np.sum(pred == y_test)}/{y_test.shape[0]}")

