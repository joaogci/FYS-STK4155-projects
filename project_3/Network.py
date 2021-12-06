import numpy as np
from time import time
from typing import Callable

class NeuralNetwork():
    """
        Neural Network Class
    """    
    
    def __init__(self, n_input_nodes: int, random_state: int = int(time())):
        """
            Initiates Neural Network. 
            The number of input nodes has to be the number of features. Then an array can be passed contaning all of the datapoints for that feature.
            Parameters:
                n_input_nodes (int): number of input nodes
                random_state (int): seed for rng
        """
        self.random_state = random_state
        self.rng = np.random.default_rng(np.random.MT19937(seed=random_state))

        self.layers = list()
        self.n_input_nodes = n_input_nodes

    def add_layer(self, n_nodes: int, activation_function: tuple[callable, callable]):
        """
            Adds a layer to the NeuralNetwork.
            Parameters:
                n_nodes (int): number of nodes for the given layer
                activation_function (tuple[callable, callable]): tuple containing two Callable objects. The first is the activation function and the second its derivative.
        """
        if len(self.layers) == 0:
            self.layers.append(Layer(n_nodes, self.n_input_nodes, activation_function, self.rng))
        else:
            self.layers.append(Layer(n_nodes, self.layers[-1].get_nodes(), activation_function, self.rng))

        self.n_layers = len(self.layers)

    def predict(self, inputs: np.ndarray):
        """
            FeedForward the given inputs. Gives a prediction for the given weights and biases.
            Parameters:
                inputs (np.ndarray): inputs for the feedforward step.
        """
        a_l = inputs

        for layer in self.layers:
            z_l = layer.weights @ a_l + layer.biases
            a_l = layer.act_function(z_l)

        return a_l

    def feed_forward(self, inputs: np.ndarray):
        """
            FeedForward for traning.
            Parameters:
                inputs (np.ndarray): inputs fot the feedforward step.
        """
        a_l = list([inputs])
        z_l = list()

        for layer in self.layers:
            z_l.append(layer.weights @ a_l[-1] + layer.biases)
            a_l.append(layer.act_function(z_l[-1]))

        return a_l, z_l

    def back_propagation(self, a: np.ndarray, z: np.ndarray, target: np.ndarray, 
                         grad_C: Callable):
        """
            BackPropagation for the training of our neural network.
            Parameters:
                a (np.ndarray): output of the activation functions for every layer.
                z (np.ndarray): input for the activation functions for every layer.
                target (np.ndarray): target values for training.
                grad_C (Callable): gradient of the CostFunction with respect to the data points. 
        """
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
        """
            Training function for the NeuralNetwork. Trains the weights and biases with Stochastic Gradient Descent.
            Parameters:
                inputs (np.ndarray): inputs to the NeuralNetwork.
                target (np.ndarray): targets for the NeuralNetwork.
                grad_C (Callable): gradient of the CostFunction with respect to the data points.
                epochs (int): epochs for the training process.
                leraning_rate (Callable): learning rate for the SGD method. Can be a function of the epochs.
                size_batches (int): size of the minibatches.
                regularization (float): l2 regularization parameter.
        """
        for epoch in range(1, epochs + 1):
            perm = self.rng.permutation(inputs.shape[1])
            inputs = inputs[:, perm]
            target = target[:, perm]
            
            for i in range(target.shape[1] // size_batches):
                index = np.arange(i*size_batches, (i+1)*size_batches, 1)
                
                a, z = self.feed_forward(inputs[:, index])
                grad_C_w, grad_C_b = self.back_propagation(a, z, target[:, index], grad_C)
                
                for l, layer in enumerate(self.layers):
                    layer.biases = layer.biases - learning_rate(epoch) * (grad_C_b[l] + regularization * layer.biases)
                    layer.weights = layer.weights - learning_rate(epoch) * (grad_C_w[l] + regularization * layer.weights)
                    
            print(f" [ epoch: {epoch}/{epochs} ] ", end='\r')

class Layer():
    """
    Layer class for keeping track of weights, biases and activation function for layers in Feed Forward Neural Network.
    """
    
    def __init__(self, n_nodes: int, n_nodes_prev_layer: int, 
                activation_function: tuple[Callable, Callable], rng: np.random.Generator,
                bias_init: float = 1e-3):
        """
        Initialises the layer with an activation function, weight matrix, and bias vector.
        Parameters:
            n_nodes (int): number of wanted nodes in the layer.
            n_nodes_prev_layer (int): number of nodes in previous layer.
            activation_function (tuple[Callable, Callable]): tuple containing wanted activation function and it's derivative.
            rng (np.random.Generator): psuedo random number generator.
            bias_init (float): value for the initialization of the bias.
        """
        self.n_nodes = n_nodes
        self.n_nodes_prev_layer = n_nodes_prev_layer
        
        self.act_function = activation_function[0]
        self.d_act_function = activation_function[1]
        
        self.weights = rng.normal(0, 1, size=(self.n_nodes, self.n_nodes_prev_layer)) / self.n_nodes
        self.biases = rng.uniform(0, 1, size=(self.n_nodes, 1)) * bias_init
        
    def get_nodes(self):
        """
            Returns the number of nodes in the layer.
        """
        return self.n_nodes

