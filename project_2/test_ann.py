
import numpy as np
from NeuralNetwork.activation.Sigmoid import Sigmoid
from NeuralNetwork.activation.Linear import Linear
from NeuralNetwork.HiddenLayer import HiddenLayer
from NeuralNetwork.OutputLayer import OutputLayer
from NeuralNetwork.Model import Model

model = Model(5, random_state=0)

model.add_layer(HiddenLayer(7, Sigmoid()))
model.add_layer(OutputLayer(3, Sigmoid()))

outputs = model.feed_forward([0, 0, 0, 1, 0])
print(outputs)
