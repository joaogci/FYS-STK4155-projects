# Feed-forward neural network implementation & comparisons

This directory contains an implementation of a feed-forward neural network, with several test cases to compare it to common linear and logistic regression methods, for use for both regression and classification problems. The aim is to apply the ANN to the [Wisconsin breast cancer data](https://www.kaggle.com/uciml/breast-cancer-wisconsin-data).

## Getting started

The [NeuralNetwork](NeuralNetwork) directory contains the actual implementation of the ANN. To use it in a script, the central model class, hidden and output layer classes, at least one cost function class, and any number of activation function subclasses should be imported, as such:
```py
from NeuralNetwork.Model import Model # The core Neural Network class
from NeuralNetwork.Layer import HiddenLayer, OutputLayer
from NeuralNetwork.ActivationFunctions import ReLU, Sigmoid, Tanh, Softmax # The desired activation functions to use; see the ActivationFunctions.py class for the full list
from NeuralNetwork.cost_function.LinearRegression import LinearRegression # For regression problems
from NeuralNetwork.cost_function.LogisticRegression import LogisticRegression # For classification problems
```

From there, it's easy to instantiate the `Model` class and add layers to it:
```py
input_size = 5 # Depends on the problem at hand
output_size = 3
nn = Model(input_size, cost_function=LogisticRegression(...)) # Init model
nn.add_layer(HiddenLayer(100, activation_function=Sigmoid()))
nn.add_layer(HiddenLayer(60, activation_function=Tanh()))
# ... and so on, for any number of hidden layers
nn.add_layer(OutputLayer(3, activation_function=Softmax())) # For logistic regression, the activation function should be Softmax or Sigmoid; for linear regression, it should be Linear
```

Training the model can be done by feeding it appropriately-sized inputs and targets:
```py
learning_rate = 0.01
sgd = True # Whether to use stochastic gradient descent
iterations = 1000 # Number of epochs to train for
minibatch_size = 5 # Only used for SGD
lmbda = 1e-6 # Regularization term (optional)
nn.train(inputs, targets, learning_rate, sgd=sgd, epochs=iterations, minibatch_size=minibatch_size, regularization=lbmda)
```

An adaptive learning rate can be used instead of a constant learning rate by setting `final_learning_rate` to some value (typically lower than `initial_learning_rate`) in the `train` call.

Additionally, if wanting to instead grid-search the best result out of a combination of two of the hyper-parameter, the `grid_train` method can be called instead, by feeding two of the above parameters as lists or numpy arrays, which the model will go over and re-train for each of the elements, either dumping the results to a `pickle` file or plotting the data in a contour plot directly, depending on the settings used.
