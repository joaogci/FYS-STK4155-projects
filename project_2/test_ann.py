
from NeuralNerwork.Sigmoid import Sigmoid
from NeuralNerwork.HiddenLayer import HiddenLayer
from NeuralNerwork.OutputLayer import OutputLayer
from NeuralNerwork.Model import Model

model = Model(0)

activation = Sigmoid()
layer = HiddenLayer(activation)
model.add_layer(layer)
