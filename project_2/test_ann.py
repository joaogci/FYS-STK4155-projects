
import numpy as np
from NeuralNetwork.activation.Sigmoid import Sigmoid
from NeuralNetwork.activation.Linear import Linear
from NeuralNetwork.HiddenLayer import HiddenLayer
from NeuralNetwork.OutputLayer import OutputLayer
from NeuralNetwork.Model import Model

# Settings
train_iterations = 250
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
model.add_layer(HiddenLayer(100, Sigmoid()))
model.add_layer(OutputLayer(3, Sigmoid())) # <- should probably be using softmax instead for the output layer in classification problems!

# Print initial outputs
print('\nBefore training:')
outputs = model.feed_forward(inputs)
# To run with a single input set instead of several, the following is possible:
# outputs = model.feed_forward([0, 0])
print(outputs.round(1))

# Train network
print()
for i in range(train_iterations):
    print(int(i / train_iterations * 100), '%', end='\r')
    for j in range(len(targets)): # For now, back_prop only deals with these one row at a time
        model.back_prop(inputs[j], targets[j], learning_rate=0.1)
print('100%')

# Print final outputs
print('\nAfter training:')
print(model.feed_forward(inputs).round(1))
print('\nTargets:')
print(targets)
