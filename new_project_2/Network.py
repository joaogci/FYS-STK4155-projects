import numpy as np
from time import time
from Layer import Layer

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

    def predict(self, inputs: np.matrix):
        a_l = inputs

        for layer in self.layers:
            z_l = layer.weights @ a_l + layer.biases
            a_l = layer.sigma(z_l)

        return a_l

    def feed_forward(self, inputs: np.matrix):
        a_l = list([inputs])
        z_l = list()

        for layer in self.layers:
            z_l.append(layer.weights @ a_l[-1] + layer.biases)
            a_l.append(layer.sigma(z_l[-1]))

        return a_l, z_l

    def back_propagation(self, inputs: np.matrix, target: np.matrix, grad_cost_function):
        a, z = self.feed_forward(inputs)

        grad_C_w = list()
        grad_C_b = list()

        grad_sigma_L = self.layers[-1].grad_sigma
        delta = np.multiply(grad_cost_function(target, a[-1]), grad_sigma_L(z[-1]))
        grad_C_w.append(delta @ a[-2].T)
        grad_C_b.append(np.sum(delta, axis=1, keepdims=True))

        for l in range(2,self.n_layers+1):
            grad_sigma_l = self.layers[- l].grad_sigma
            weights_next = self.layers[- l + 1].weights
            delta = np.multiply((weights_next.T @ delta), grad_sigma_l(z[- l]))
            grad_C_w.append(delta @ a[- l - 1].T)
            grad_C_b.append(np.sum(delta, axis=1, keepdims=True))

        grad_C_w.reverse()
        grad_C_b.reverse()

        for l, layer in enumerate(self.layers):
            layer.biases = layer.biases - 0.1 * grad_C_b[l]
            layer.weights = layer.weights - 0.1 * grad_C_w[l]
