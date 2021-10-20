
import numpy as np
from time import time

from .Layer import Layer, HiddenLayer, OutputLayer

class Model:
    
    def __init__(self, input_size: int, random_state: int = int(time())):
        """
            Artificial neural network class
            Parameters:
                input_size (int): Size of the input layer (i.e. number of features), which will determine the number of weights in the first hidden layer
                random_state (int): Seed value to use for RNG
        """

        self._input_size = input_size

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
        if isinstance(layer, OutputLayer) and len(self.layers) <= 0:
            print('\033[91mCannot add an output layer before adding at least one hidden layer! Make sure layers are being added in the correct order.\033[0m')
            return

        # Compute number of inputs the layer will be receiving to init weights
        n_inputs = self._input_size
        if len(self.layers) > 0:
            n_inputs = self.layers[len(self.layers) - 1].get_size()
        layer.init_weights(n_inputs, self.rng)
        
        # Add layer
        if isinstance(layer, OutputLayer):
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



    def feed_forward(self, inputs: np.matrix, training: bool = False) -> tuple:
        """
            Runs through the network once with a given list of inputs and returns the obtained outputs
            Parameters:
                inputs (np.matrix|list<float>): The set of inputs to give to the network
                training (bool): If true, will return hidden layer activations alongside the actual outputs
            Returns:
                (np.matrix|list<float>): Outputs obtained out of the output layer after running through all layers
                (list<np.matrix>): Hidden layer activation outputs, returned only if `training` was `true`
        """

        # If the input is given as a 1D array, we're wanting to use that as a single row in the input matrix (i.e. run with a single set of input data)
        output_list = False
        if isinstance(inputs, list):
            inputs = np.matrix([inputs])
            output_list = True

        if not self.is_ready():
            print('\033[91mNetwork hasn\'t been given an output layer! Make sure the neural network is set-up with all layers before starting training\033[0m')
            return

        if inputs.shape[1] != self._input_size:
            print('\033[91mCannot feed input of size', inputs.shape, 'into ANN with input size', self._input_size, '.\033[0m')
            return

        # Process from layer to layer sequentially, passing the output of each layer into the next
        tmp = inputs
        if training:
            a_h = []
        for layer in self.layers:

            # Activate the layer
            tmp = layer.forward(tmp)

            if tmp is None:
                print('\033[91mLayer gave invalid results; see above for details regarding the error.\033[0m')
                return None
            
            # In training, keep track of hidden layer outputs
            if training and isinstance(layer, HiddenLayer):
                a_h.append(np.matrix(tmp))
        
        # Output of final layer = output of network
        if output_list:
            tmp = tmp[0] # Output as a list if the input was given as such
        if training:
            return tmp, a_h
        return tmp



    def back_prop(self, inputs: np.matrix, targets: np.matrix, learning_rate: float = 0.1):
        """
            Back-propagates once with a set of actual and desired outputs, so the next run will match the targets closer (hopefully)
            Parameters:
                inputs (np.matrix): Inputs to train for
                targets (np.matrix): Desired outcome values
                learning_rate (float): Learning rate η to use to update the weights & biases
        """

        # For now, each training set needs to be fed one by one
        # @todo Remove this check and implement passing all training sets at once, much like feed_forward
        if inputs.shape[0] != 1:
            print('\033[91mERROR: for now, back_prop only supports passing in ONE input set at a time')
            print('This will be extended to supporting an array of input & output sets instead of one at a time later.\033[0m')
            exit()

        if not self.is_ready():
            print('\033[91mNetwork hasn\'t been given an output layer! Make sure the neural network is set-up with all layers before starting training\033[0m')
            return

        # Feed forward once to obtain outputs
        outputs, a_h = self.feed_forward(inputs, training=True)

        # Dimensionality check
        if outputs.shape != targets.shape or outputs.shape[1] != self.layers[len(self.layers) - 1].get_size():
            print('\033[91mMismatching outputs/targets size; should be (x,', self.layers[len(self.layers) - 1].get_size(), '), got', outputs.shape, 'and', targets.shape, 'instead..\033[0m')
            return
        
        # Compute errors & gradient descent for each layer
        # Going backwards from last to first layer
        prev_layer_err = outputs - targets
        for i in range(len(self.layers)-1, -1, -1):

            # In the first layer, the input is just straight-up the data
            layer_in = a_h[i-1] if i > 0 else inputs

            # Update layer
            prev_layer_err = self.layers[i].backward(layer_in, prev_layer_err, learning_rate)
        
