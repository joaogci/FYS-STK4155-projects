import numpy as np
from time import time

from Layer import Layer

class Model:
    
    def __init__(self, random_state: int = int(time())):
        """
            Model class...
        """
        self.random_state = random_state
        self.rng = np.random.default_rng(np.random.MT19937(seed=self.random_state))
        
        self.layers = list()
        self._has_hidden = False
        
    def add_layer(self, layer: Layer):
        """
            Adds a layer to the Model class. 
            It has to have at least one HiddenLayer and only one OutputLayer.
            The layers must be added by the order they act. 
        """
        
        # add layers
        if layer.NAME == "OUTPUT" and not self._has_hidden:
            self.layers.append(layer)
            self._has_hidden = True
        else:
            self.layers.append(layer)
    
    





