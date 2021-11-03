
import numpy as np
from abc import ABC
from .activation.ActivationFunction import ActivationFunction

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
        self._activationFn = activation_function
        self._weights = None
        self._biases = np.ones((1, self._size)) * initial_bias

    def get_size(self) -> int:
        """
            Returns the size of the layer
            Returns:
                (int): Size of the layer (i.e. number of neurons)
        """
        return self._size

    def init_weights(self, size: int, rng: np.random.Generator):
        """
            Initialises the weights array for the layer with stochastic noise
            The size corresponding to the number of nodes in the previous layer
            Parameters:
                size (int): Number of inputs the layer will be receiving, i.e. number of nodes in the previous layer
                rng (np.random.Generator): Random number generator to use when selecting initial weights
        """
        self._weights = rng.uniform(-1, 1, (self._size, size))

    def forward(self, inputs: np.matrix) -> np.matrix:
        """
            Runs through the layer once with a given list of inputs and returns the outputs obtained
            Parameters:
                inputs (np.matrix): Inputs to run through the nodes (must match size of weights/biases!)
            Returns:
                (np.matrix): Outputs from the different nodes - size corresponds to size of the layer
        """
        
        if self._weights is None or self._biases is None or len(self._weights) != self._biases.shape[1]:
            print('\033[91mLayer hasn\'t been assigned weights and biases! Ensure init_weights_and_biases is called before starting.\033[0m')
            return None

        if inputs.shape[1] != len(self._weights[0]):
            print('\033[91mLayer hasn\'t been initialised with the right amount of weights/biases! Expected', len(self._weights[0]), 'inputs, given', inputs.shape, '!\033[0m')
            return None
        
        # Accumulate inputs for each node
        outputs = np.zeros((inputs.shape[0], self._size))
        sums = np.zeros((inputs.shape[0], self._size))
        for input_idx in range(inputs.shape[0]):
            for i in range(self._size):
                # a'_i = σ( Σ_j( a_j * w_ij + b_ij ) )
                for j in range(self._weights.shape[1]):
                    sums[input_idx, i] += inputs[input_idx, j] * self._weights[i, j] + self._biases[0, i]
        outputs = self._activationFn(sums)

        return outputs

    def backward(self, inputs: np.matrix, error: np.matrix, learning_rate: float, lmbda: float) -> np.matrix:
        """
            Gradient descent to optimize the layer
            Parameters:
                inputs (np.matrix): The inputs the layer receives
                error (np.matrix): Computed error estimate for the layer
                learning_rate (float): Learning rate η to use to update the weights & biases
                lmbda (float): Hyperparameter to control the rate of descent
            Returns:
                (np.matrix): Weighted error in inputs, to use to train the previous layer
        """
        # Compute gradients
        # Simple gradient descent
        weights_gradient = inputs.T @ error
        bias_gradient = np.sum(error, axis=0)

        if lmbda > 0.0:
            weights_gradient += lmbda * self._weights.T

        # Adjust weights and biases
        self._weights -= learning_rate * weights_gradient.T
        self._biases -= learning_rate * bias_gradient

        # Return the estimated error in inputs
        weighted_err = np.matmul(error, self._weights)
        return np.multiply(weighted_err, np.multiply(inputs, (1.0 - inputs)))



# Declare Hidden and Output layer classes for nicer syntax
class HiddenLayer(Layer):
    ...

class OutputLayer(Layer):
    ...
