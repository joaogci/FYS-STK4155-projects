
import numpy as np
from abc import ABC
from .activation_function.ActivationFunction import ActivationFunction

class Layer(ABC):
    """
        Abstract class that can be inherited to define two different types of Layers (hidden and output).
    """

    def __init__(self, size: int, activation_function: ActivationFunction, initial_bias: float = 1e-3):
        """
            Initialises the layer with a custom activation function
            Parameters:
                size (int): Number of nodes in the hidden layer
                activation_function (ActivationFunction): The activation function to use for the layer
                initial_bias (float): Initial value to initialize the bias to, typically zero or a small value (see https://cs231n.github.io/neural-networks-2/)
        """
        self._size = size
        self._activation_fn = activation_function
        self._weights = None
        self._biases = np.ones((self._size, 1)) * initial_bias

    def get_size(self) -> int:
        """
            Returns the size of the layer
            Returns:
                (int): Size of the layer (i.e. number of neurons)
        """
        return self._size

    def init_weights(self, input_size: int, rng: np.random.Generator):
        """
            Initialises the weights array for the layer with stochastic noise
            The size corresponding to the number of nodes in the previous layer
            Parameters:
                input_size (int): Number of inputs the layer will be receiving, i.e. number of nodes in the previous layer
                rng (np.random.Generator): Random number generator to use when selecting initial weights
        """
        self._weights = rng.uniform(-1, 1, (self._size, input_size))

    def forward(self, inputs: np.matrix) -> tuple:
        """
            Runs through the layer once with a given list of inputs and returns the outputs obtained
            Parameters:
                inputs (np.matrix): Inputs to run through the nodes (must match size of weights/biases!)
            Returns:
                (np.matrix): Outputs from the different nodes - size corresponds to size of the layer
                (np.matrix): Same outputs without activation function
        """
        
        if self._weights is None or self._biases is None or len(self._weights) != self._biases.shape[0]:
            print('\033[91mLayer hasn\'t been assigned weights! Ensure init_weights is called before starting.\033[0m')
            return None, None

        if inputs.shape[1] != len(self._weights[0]):
            print('\033[91mLayer hasn\'t been initialised with the right amount of weights/biases! Expected', len(self._weights[0]), 'inputs, given', inputs.shape, '!\033[0m')
            return None, None
        
        # Accumulate inputs for each node
        # Because of the way we structure the input, we need to transpose inputs and outputs :)
        z = (self._weights @ inputs.T + self._biases).T
        return self._activation_fn(z), z

    def backward(self, activated_inputs: np.matrix, inputs: np.matrix, error: np.matrix, learning_rate: float, regularization: float) -> np.matrix:
        """
            Gradient descent to optimize the layer
            Parameters:
                activated_inputs (np.matrix): The inputs the layer receives
                inputs (np.matrix): The inputs the layer receives (no activation fn)
                error (np.matrix): Computed error estimate for the layer
                learning_rate (float): Learning rate η to use to update the weights & biases
                regularization (float): Regularization parameter λ to control the rate of descent
            Returns:
                (np.matrix): Weighted error in inputs, to use to train the previous layer
        """
        # Compute gradients
        # Simple gradient descent
        weights_gradient = activated_inputs.T @ error
        bias_gradient = error

        # Add regularization term to weights gradient (might be 0)
        weights_gradient += regularization * self._weights.T

        # Adjust weights and biases
        self._weights -= learning_rate * weights_gradient.T
        self._biases -= learning_rate * bias_gradient.T

        # Return the estimated error in inputs
        return np.multiply((error @ self._weights), self._activation_fn.d(inputs))



# Declare Hidden and Output layer classes for nicer syntax
# The only difference between the two is the name
class HiddenLayer(Layer):
    ...

class OutputLayer(Layer):
    ...
