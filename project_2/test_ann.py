
import numpy as np
import matplotlib.pyplot as plt
from NeuralNetwork.activation_function.Sigmoid import Sigmoid
from NeuralNetwork.activation_function.Linear import Linear
from NeuralNetwork.activation_function.Softmax import Softmax
from NeuralNetwork.Layer import HiddenLayer, OutputLayer
from NeuralNetwork.Model import Model
from NeuralNetwork.cost_function.LinearRegression import LinearRegression


# Settings
learning_rate = 0.5
lmbda = 8e-4
plot_from = 100 # Train iteration at which to start plotting MSEs
train_iterations = 2000 # Max training iteration

# Train for XOR, AND, OR as a test
inputs = np.matrix([
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1]
])
targets = np.matrix([ # First column: ^, second column: &, third column: |
    [0, 0, 0],
    [1, 0, 1],
    [1, 0, 1],
    [0, 1, 1]
])


# Init network + layers
model = Model(2, random_state=0, cost_function=LinearRegression(inputs, targets, inputs, targets))
model.add_layer(HiddenLayer(60, Sigmoid()))
model.add_layer(OutputLayer(3, Sigmoid()))

# Print initial outputs
print('\nBefore training:')
outputs = model.feed_forward(inputs)
# To run with a single input set instead of several, the following is possible:
# outputs = model.feed_forward([0, 0])
print(outputs.round(5))

# Train network
model.train(inputs, targets, epochs=train_iterations, learning_rate=learning_rate, regularization=lmbda)

# Print final outputs
print('\nAfter training:')
results = model.feed_forward(inputs)
print(results.round(5))
print('\nTargets:')
print(targets)

print(model.fwd_mse(inputs, targets))
