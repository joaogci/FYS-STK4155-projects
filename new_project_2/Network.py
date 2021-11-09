import numpy as np
from time import time
from typing import Callable

class NeuralNetwork():

    def __init__(self, n_input_nodes: int, random_state: int = int(time())):
        self.random_state = random_state
        self.rng = np.random.default_rng(np.random.MT19937(seed=random_state))

        self.layers = list()
        self.n_input_nodes = n_input_nodes

    def add_layer(self, n_nodes: int, activation_function: tuple):
        if len(self.layers) == 0:
            self.layers.append(Layer(n_nodes, self.n_input_nodes, activation_function, self.rng))
        else:
            self.layers.append(Layer(n_nodes, self.layers[-1].get_nodes(), activation_function, self.rng))

        self.n_layers = len(self.layers)

    def predict(self, inputs: np.ndarray):
        a_l = inputs

        for layer in self.layers:
            z_l = layer.weights @ a_l + layer.biases
            a_l = layer.act_function(z_l)

        return a_l

    def feed_forward(self, inputs: np.ndarray):
        a_l = list([inputs])
        z_l = list()

        for layer in self.layers:
            z_l.append(layer.weights @ a_l[-1] + layer.biases)
            a_l.append(layer.act_function(z_l[-1]))

        return a_l, z_l

    def back_propagation(self, a: np.ndarray, z: np.ndarray, target: np.ndarray, 
                         grad_C: Callable):
        grad_C_w = list()
        grad_C_b = list()

        grad_sigma_L = self.layers[- 1].d_act_function
        delta = np.multiply(grad_C(target, a[- 1]), grad_sigma_L(z[- 1]))
        grad_C_w.append(delta @ a[- 2].T)
        grad_C_b.append(np.sum(delta, axis=1, keepdims=True))
        
        for l in range(2,self.n_layers+1):
            grad_sigma_l = self.layers[- l].d_act_function
            weights_next = self.layers[- l + 1].weights
            delta = np.multiply((weights_next.T @ delta), grad_sigma_l(z[- l]))
            grad_C_w.append(delta @ a[- l - 1].T)
            grad_C_b.append(np.sum(delta, axis=1, keepdims=True))

        grad_C_w.reverse()
        grad_C_b.reverse()
        return grad_C_w, grad_C_b

    def train(self, inputs: np.ndarray, target: np.ndarray, grad_C: Callable, 
              epochs: int, learning_rate: Callable, size_batches: int,
              regularization: float = 0):
        
        for epoch in range(1, epochs + 1):
            perm = self.rng.permutation(inputs.shape[1])
            inputs = inputs[:, perm]
            target = target[:, perm]
            
            for i in range(target.shape[1] // size_batches):
                index = np.arange(i*size_batches, (i+1)*size_batches, 1)
                
                a, z = self.feed_forward(inputs[:, index])
                grad_C_w, grad_C_b = self.back_propagation(a, z, target[:, index], grad_C)
                
                for l, layer in enumerate(self.layers):
                    layer.biases = layer.biases - learning_rate(epoch) * (grad_C_b[l] + regularization * self.layers[l].biases)
                    layer.weights = layer.weights - learning_rate(epoch) * (grad_C_w[l] + regularization * self.layers[l].weights)
            
            print(f" [ epoch: {epoch}/{epochs} ] ", end='\r')
        

class Layer():
    """
    Layer class for keeping track of weights, biases and activation
    function for layers in Feed Forward Neural Network.
    """
    
    def __init__(self, n_nodes: int, n_nodes_prev_layer: int, 
                activation_function: tuple[Callable], rng: np.random.Generator):
        """
        Initialises the layer with an activation function, weight 
        matrix, and bias vector.
        Parameters:
            n_nodes (int): Number of wanted nodes in the layer
            n_nodes_prev_layer (int): Number of nodes in previous layer
            activation_function (tuple[Callable]): Tuple containing wanted 
                                         activation function and it's
                                         derivative.
            rng (np.random.Generator): Psuedo random number generator
        """
        self.n_nodes = n_nodes
        self.n_nodes_prev_layer = n_nodes_prev_layer
        
        self.act_function = activation_function[0]
        self.d_act_function = activation_function[1]
        
        self.weights = rng.normal(0, 1, size=(self.n_nodes, self.n_nodes_prev_layer))
        self.biases = rng.uniform(0, 1, size=(self.n_nodes, 1)) * 1e-3 # need to change later
        
    def get_nodes(self):
        return self.n_nodes

