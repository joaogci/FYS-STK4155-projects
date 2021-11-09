import numpy as np

class Layer():
    
    def __init__(self, n_nodes: int, n_nodes_prev_layer: int, activation_function: tuple, rng: np.random.Generator):
        self.n_nodes = n_nodes
        self.n_nodes_prev_layer = n_nodes_prev_layer
        
        self.sigma = activation_function[0]
        self.grad_sigma = activation_function[1]
        
        self.weights = rng.normal(0, 1, size=(self.n_nodes, self.n_nodes_prev_layer))
        self.biases = rng.uniform(0, 1, size=(self.n_nodes, 1)) * 1e-3 # need to change later
        
    def get_nodes(self):
        return self.n_nodes
    
        
    