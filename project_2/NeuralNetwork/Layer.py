
import numpy as np
from abc import ABC, abstractmethod
from .activation.ActivationFunction import ActivationFunction

class Layer(ABC):
    """
        Abstract class that can be inherited to define two different types of Layers (hidden and output).
    """
    
    NAME = ""

    def __init__(self, size: int, activation_function: ActivationFunction):
        """
            Initialises the layer with a custom activation function
            Parameters:
                size (int): Number of nodes in the hidden layer
                activation_function (ActivationFunction): The activation function to use for the layer
        """
        self._size = size
        self._activationFn = activation_function
        self._weights = None
        self._biases = None

    def get_size(self) -> int:
        """
            Returns the size of the layer
            Returns:
                (int): Size of the layer (i.e. number of neurons)
        """
        return self._size

    def init_weights_and_biases(self, size: int, rng: np.random.Generator):
        """
            Initialises the weights array for the layer with stochastic noise, and the biases with zeros (see https://cs231n.github.io/neural-networks-2/)
            The size corresponding to the number of nodes in the previous layer
            Parameters:
                size (int): Number of inputs the layer will be receiving, i.e. number of nodes in the previous layer
                rng (np.random.Generator): Random number generator to use when selecting initial weights
        """
        self._weights = rng.uniform(0, 1, (self._size, size))
        self._biases = np.zeros((self._size, size))

    def forward(self, inputs: np.matrix) -> np.matrix:
        """
            Runs through the layer once with a given list of inputs and returns the outputs obtained
            Parameters:
                inputs (np.matrix): Inputs to run through the nodes (must match size of weights/biases!)
            Returns:
                (np.matrix): Outputs from the different nodes - size corresponds to size of the layer
        """
        
        if self._weights is None or self._biases is None or len(self._weights) != len(self._biases):
            print('\033[91mLayer hasn\'t been assigned weights and biases! Ensure init_weights_and_biases is called before starting.\033[0m')
            return None

        if inputs.shape[1] != len(self._weights[0]):
            print('\033[91mLayer hasn\'t been initialised with the right amount of weights/biases! Expected', len(self._weights[0]), 'inputs, given', inputs.shape, '!\033[0m')
            return None
        
        # Accumulate inputs for each node
        outputs = np.zeros((inputs.shape[0], self._size))
        for input_idx in range(inputs.shape[0]):
            for i in range(self._size):
                # a'_i = σ( Σ_j( a_j * w_ij + b_ij ) )
                sum = 0
                for j in range(self._weights.shape[1]):
                    sum += inputs[input_idx, j] * self._weights[i, j] + self._biases[i, j]
                outputs[input_idx, i] = self._activationFn(sum)

        return outputs
