import numpy as np
from time import time

from .Layer import Layer

class Model:
    
    def __init__(self, random_state: int = int(time())):
        """
            Model class...
        """
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
        
        # Add layer
        if layer.NAME == "OUTPUT" and not self._has_output:
            self.layers.append(layer)
            self._has_output = True # Locks the layers array to prevent adding more after the output layer
        else:
            self.layers.append(layer)
    
    





