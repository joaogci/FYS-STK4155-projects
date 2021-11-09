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

def test_binary():
    """
    Test if we can reproduce the XOR, OR and AND binary gate.
    """
    inputs = np.matrix([
        [0, 0, 1, 1],
        [0, 1, 0, 1]
    ])
    targets = np.matrix([ 
        [0, 1, 1, 0],   # xor
        [0, 1, 1, 1],   # or
        [0, 0, 0, 1]    # and
    ])

    neural_network = NeuralNetwork(2, random_state=seed)
    neural_network.add_layer(16, (sigmoid, grad_sigmoid))
    neural_network.add_layer(3, (sigmoid, grad_sigmoid))

    for i in range(1, 1000+1):
        print(f"{i}/1000")
        neural_network.back_propagation(inputs, targets, grad_cross_ent)

    output = neural_network.predict(inputs)
    print(output)
    print(output.round())

def test_breast_cancer():
    """
    Test if we can determine the type of tumor for breast cancer data.
    """
    # Extract data for ScikitLearn
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

    pred = neural_network.predict(X_test)
    pred = pred.round()

    print(f"accuracy: {np.sum(pred == y_test)}/{y_test.shape[0]}")

def test_mnist():
    """
    Test if we can determine handwritten numbers from MNist dataset.
    """
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

    neural_network = NeuralNetwork(64, random_state=seed)
    neural_network.add_layer(20, (sigmoid, grad_sigmoid))
    neural_network.add_layer(10, (sigmoid, grad_sigmoid))

    

    for i in range(1, 250+1):
        print(f"{i}/250")
        neural_network.back_propagation(X_train, y_train, grad_cross_ent)

    pred = neural_network.predict(X_test)
    pred = np.argmax(pred, axis=0)

    print(f"accuracy: {np.sum(pred == y_test)}/{y_test.shape[0]}")

# test_binary()
# test_breast_cancer()
test_mnist()