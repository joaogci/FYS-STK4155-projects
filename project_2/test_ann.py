
import numpy as np
import matplotlib.pyplot as plt
from NeuralNetwork.activation.Sigmoid import Sigmoid
from NeuralNetwork.activation.Linear import Linear
from NeuralNetwork.activation.Softmax import Softmax
from NeuralNetwork.Layer import HiddenLayer, OutputLayer
from NeuralNetwork.Model import Model


# Settings
learning_rate = 0.12
lmbda = 8e-4
plot_from = 250 # Train iteration at which to start plotting MSEs
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
model = Model(2, random_state=0)
model.add_layer(HiddenLayer(60, Sigmoid()))
model.add_layer(OutputLayer(3, Sigmoid()))

# Print initial outputs
print('\nBefore training:')
outputs = model.feed_forward(inputs)
# To run with a single input set instead of several, the following is possible:
# outputs = model.feed_forward([0, 0])
print(outputs.round(5))

# Train network
print()
mses = np.zeros(train_iterations-plot_from)
for i in range(train_iterations):
    print(int(i / train_iterations * 100), '%', end='\r')
    model.back_prop(inputs, targets, learning_rate=learning_rate, regularization=lmbda)
    if i >= plot_from:
        mses[i-plot_from] = model.fwd_mse(inputs, targets)
print('100%')

plt.plot(range(plot_from, train_iterations), mses)
plt.ylabel('MSE')
plt.xlabel('backprop iterations')
plt.title(fr'MSE as a function of backpropagation iterations, $\eta={learning_rate}$, $\lambda={lmbda}$')

# Print final outputs
print('\nAfter training:')
results = model.feed_forward(inputs)
print(results.round(5))
print('\nTargets:')
print(targets)

plt.show()