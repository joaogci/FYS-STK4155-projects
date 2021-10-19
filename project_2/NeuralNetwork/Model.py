
import numpy as np
from time import time

from .Layer import Layer

class Model:
    
    def __init__(self, input_size: int, learning_rate: float = 0.1, random_state: int = int(time())):
        """
            Artificial neural network class
            Parameters:
                input_size (int): Size of the input layer, which will determine the number of weights in the first hidden layer
                learning_rate (float): η learning rate to apply when training
                random_state (int): Seed value to use for RNG
        """

        self._input_size = input_size
        self._learning_rate = learning_rate

        self.random_state = random_state
        self.rng = np.random.default_rng(np.random.MT19937(seed=self.random_state))
        
        self.layers = list()
        self._has_output = False
        
    
    def add_layer(self, layer: Layer):
        """
            Adds a layer to the Model class. 
            It has to have at least one HiddenLayer and only one OutputLayer.
            The layers must be added by the order they act. 
        """

        # Ensure no more than 1 output layer
        if self._has_output:
            print('\033[91mCannot add another layer after the network\'s output layer! Make sure layers are being added in the correct order.\033[0m')
            return
        
        # Ensure at least one hidden layer
        if layer.NAME == "OUTPUT" and len(self.layers) <= 0:
            print('\033[91mCannot add an output layer before adding at least one hidden layer! Make sure layers are being added in the correct order.\033[0m')
            return

        # Compute number of inputs the layer will be receiving to init weights
        n_inputs = self._input_size
        if len(self.layers) > 0:
            n_inputs = self.layers[len(self.layers) - 1].get_size()
        layer.init_weights_and_biases(n_inputs, self.rng)
        
        # Add layer
        if layer.NAME == "OUTPUT":
            self.layers.append(layer)
            self._has_output = True # Locks the layers array to prevent adding more after the output layer
        else:
            self.layers.append(layer)

    def is_ready(self) -> bool:
        """
            Helper to determine whether the ANN is ready for training
            Returns:
                (bool): Whether the network is ready, i.e. has been given all its layers
        """
        return self._has_output



    def feed_forward(self, inputs: np.matrix) -> np.matrix:
        """
            Runs through the network once with a given list of inputs and returns the obtained outputs
            Parameters:
                inputs (np.matrix): The set of inputs to give to the network
            Returns:
                (np.matrix): Outputs obtained out of the output layer after running through all layers
        """

        if not self.is_ready():
            print('\033[91mNetwork hasn\'t been given an output layer! Make sure the neural network is set-up with all layers before starting training\033[0m')
            return

        if len(inputs) != self._input_size:
            print('\033[91mCannot feed input of size', len(inputs), 'into ANN with input size', self._input_size, '.\033[0m')
            return

        # Process from layer to layer sequentially, passing the output of each layer into the next
        tmp = inputs
        for layer in self.layers:
            tmp = layer.forward(tmp)
            if tmp is None:
                print('\033[91mLayer gave invalid results; see above for details regarding the error.\033[0m')
                return None
        
        # Output of final layer = output of network
        return tmp

