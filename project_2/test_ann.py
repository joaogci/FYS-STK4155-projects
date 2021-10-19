
import numpy as np
from NeuralNetwork.activation.Sigmoid import Sigmoid
from NeuralNetwork.activation.Linear import Linear
from NeuralNetwork.HiddenLayer import HiddenLayer
from NeuralNetwork.OutputLayer import OutputLayer
from NeuralNetwork.Model import Model

model = Model(2, random_state=0)
model.add_layer(HiddenLayer(100, Sigmoid()))
model.add_layer(OutputLayer(3, Sigmoid())) # <- should probably be using softmax instead for the output layer in classification problems!

# Train for XOR, AND, OR as a test
inputs = np.matrix([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
targets = np.matrix([
    [0, 0, 0],
    [1, 0, 1],
    [1, 0, 1],
    [0, 1, 1]
])

print('\nBefore training:')
outputs = model.feed_forward(inputs)
# To run with a single input set instead of several, the following is possible:
# outputs = model.feed_forward([0, 0])
print(outputs)


model.back_prop(outputs, targets, learning_rate=0.1)

print('\nAfter training:')
print(model.feed_forward(inputs))
