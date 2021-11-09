import numpy as np

class Layer():
    """
    Layer class for keeping track of weights, biases and activation
    function for layers in Feed Forward Neural Network.
    """
    def __init__(self, n_nodes: int, n_nodes_prev_layer: int, 
                activation_function: tuple, rng: np.random.Generator):
        """
        Initialises the layer with an activation function, weight 
        matrix, and bias vector.
        Parameters:
            n_nodes (int): Number of wanted nodes in the layer
            n_nodes_prev_layer (int): Number of nodes in previous layer
            activation_function (tuple): Tuple containing wanted 
                                         activation function and it's
                                         derivative.
            rng (np.random.Generator): Psuedo random number generator
        """
        self.n_nodes = n_nodes
        self.n_nodes_prev_layer = n_nodes_prev_layer
        
        self.sigma = activation_function[0]
        self.grad_sigma = activation_function[1]
        
        self.weights = rng.normal(0, 1, size=(self.n_nodes, self.n_nodes_prev_layer))
        self.biases = rng.uniform(0, 1, size=(self.n_nodes, 1)) * 1e-3 # need to change later
        
    def get_nodes(self):
        return self.n_nodes
    
        
    