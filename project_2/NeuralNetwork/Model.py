
import numpy as np
import matplotlib.pyplot as plt
from time import time
from typing import Callable
import pickle

from .Layer import Layer, HiddenLayer, OutputLayer
from .cost_function.CostFunction import CostFunction

from sklearn.model_selection import train_test_split


class Model:
    """
        Feed-forward neural network instance
        Can add layers to the network with add_layer
    """
    
    def __init__(self, input_size: int, cost_function: CostFunction, random_state: int = int(time())):
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
        self.cost_function = cost_function
        self._has_output = False
        
    

    def add_layer(self, layer: Layer):
        """
            Adds a layer to the Model class.
            It has to have at least one OutputLayer.
            The layers must be added by the order they act.
        """

        # Ensure no more than 1 output layer
        if self._has_output:
            print('\033[91mCannot add another layer after the network\'s output layer! Make sure layers are being added in the correct order.\033[0m')
            return

        # Compute number of inputs the layer will be receiving to init weights
        n_inputs = self._input_size
        if len(self.layers) > 0:
            n_inputs = self.layers[-1].get_size()
        layer.init_weights(n_inputs, self.rng)
        
        # Add layer
        self.layers.append(layer)
        if isinstance(layer, OutputLayer):
            self._has_output = True # Locks the layers array to prevent adding more after the output layer

    def is_ready(self) -> bool:
        """
            Helper to determine whether the ANN is ready for training
            Returns:
                (bool): Whether the network is ready, i.e. has been given all its layers
        """
        return self._has_output

    def reset(self, reset_rng: bool = True):
        """
            Resets all weights and biases in the network, to be able to train again from scratch with different parameters
            Parameters:
                reset_rng (bool): Whether to reset the random number generator, useful if using SGD to train several times in order to get the same results
        """

        if reset_rng:
            self.rng = np.random.default_rng(np.random.MT19937(seed=self.random_state))
        
        for layer in self.layers:
            layer.reset(self.rng)


    def feed_forward(self, inputs: np.matrix, training: bool = False) -> tuple:
        """
            Runs through the network once with a given list of inputs and returns the obtained outputs
            Parameters:
                inputs (np.matrix|list<float>): The set of inputs to give to the network
                training (bool): If true, will return hidden layer activations alongside the actual outputs
            Returns:
                (np.matrix|list<float>): Outputs obtained out of the output layer after running through all layers, returned only if `training` was `false`
                (list<np.matrix>): Hidden layer activated outputs, returned only if `training` was `true`
                (list<np.matrix>): Hidden layer outputs (no activation function), returned only if `training` was `true`
        """

        # If the input is given as a 1D array, we're wanting to use that as a single row in the input matrix (i.e. run with a single set of input data)
        output_list = False
        if isinstance(inputs, list):
            inputs = np.matrix([inputs])
            output_list = True
        
        if inputs.ndim == 1:
            inputs = np.matrix(inputs)

        if not self.is_ready():
            print('\033[91mNetwork hasn\'t been given an output layer! Make sure the neural network is set-up with all layers before starting training\033[0m')
            return

        if inputs.shape[1] != self._input_size:
            print('\033[91mCannot feed input of size', inputs.shape, 'into ANN with input size', self._input_size, '.\033[0m')
            return

        # Process from layer to layer sequentially, passing the output of each layer into the next
        tmp = inputs
        if training:
            a_h = [inputs]
            z_h = [inputs]
        for layer in self.layers:

            # Activate the layer
            tmp, z = layer.forward(tmp)

            # If for whatever reason some kind of error occured, the output of forward() will be null
            if tmp is None:
                print('\033[91mLayer gave invalid results; see above for details regarding the error.\033[0m')
                return None
            
            # In training, keep track of hidden layer outputs
            if training:
                z_h.append(z)
                a_h.append(tmp)
        
        # Output of final layer = output of network
        if output_list:
            tmp = tmp[0] # Output as a list if the input was given as such
        if training:
            return a_h, z_h
        return tmp

    def error(self, inputs: np.matrix, targets: np.matrix) -> float:
        """
            Feeds forward once, then returns the mean squared error between targets and outputs
            Parameters:
                inputs (np.matrix): Inputs to run the network on
                targets (np.matrix): Expected outputs
            Returns:
                (float): Mean squared error after prediction
        """
        return self.cost_function.error_nn(targets, self.feed_forward(inputs))


    def back_prop(self, inputs: np.matrix, targets: np.matrix, learning_rate: float = 0.1, regularization: float = 0):
        """
            Back-propagates once with a set of actual and desired outputs, so the next run will match the targets closer (hopefully)
            Parameters:
                inputs (np.matrix): Inputs to train for
                targets (np.matrix): Desired outcome values
                learning_rate (float): Learning rate η to use to update the weights & biases
                regularization (float): Regularization parameter λ to control rate of descent
        """

        if not self.is_ready():
            print('\033[91mNetwork hasn\'t been given an output layer! Make sure the neural network is set-up with all layers before starting training\033[0m')
            return

        # Iterate over list of inputs/targets if passing more than 1
        for i in range(inputs.shape[0]):
            ins = inputs[i]
            targs = targets[i]

            # Feed forward once to obtain outputs
            a_h, z_h = self.feed_forward(ins, training=True)

            # Dimensionality check
            if a_h[-1].shape != targs.shape or a_h[-1].shape[1] != self.layers[len(self.layers) - 1].get_size():
                print('\033[91mMismatching outputs/targets size; should be (x,', self.layers[len(self.layers) - 1].get_size(), '), got', a_h[-1].shape, 'and', targs.shape, 'instead..\033[0m')
                return
            
            # Compute errors & gradient descent for each layer
            # Going backwards from last to first layer
            prev_layer_err = np.multiply(self.cost_function.grad_C_nn(targs, a_h[-1]), self.layers[-1]._activation_fn.d(z_h[-1]))
            for j in range(len(self.layers)-1, -1, -1): # for (let i = len(self.layers) - 1; i >= 0; --i)       (python is fucking garbage)
                # Update layer
                prev_layer_err = self.layers[j].backward(a_h[j], z_h[j], prev_layer_err, learning_rate, regularization)
    

    def train(self, inputs: np.matrix, targets: np.matrix, initial_learning_rate: float = 0.1, final_learning_rate: float = None, sgd: bool = True, epochs: int = 1000, minibatch_size: int = 5, regularization: float = 0, testing_inputs: np.matrix = None, testing_targets: np.matrix = None, verbose: bool = True) -> tuple:
        """
            Back-propagates over a series of epochs using stochastic gradient descent
            Parameters:
                inputs (np.matrix): Inputs to train for
                targets (np.matrix): Desired outcome values
                initial_learning_rate (float): Learning rate at epoch = 0
                final_learning_rate (float): Learning rate at epoch = max_epochs; if passing None, will keep learning rate constant
                sgd (bool): Whether to use stochastic gradient descent or plain old gd
                epochs (int): Number of training epochs to train over
                minibatch_size (int): Size of individual mini-batches
                regularization (float): Regularization parameter λ to control rate of descent
                testing_inputs (np.matrix): If not None, will compute the error/accuracy score for the test set at each epoch
                testing_targets (np.matrix): If not None, will compute the error/accuracy score for the test set at each epoch
                verbose (bool): Whether to output the completion percentage to stdout
            Returns:
                (float): Final training error obtained by the network after the last training iteration
                (float): Final testing error obtained by the network after the last training iteration; only returned if testing_inputs and testing_targets are passed
        """

        if not self.is_ready():
            print('\033[91mNetwork hasn\'t been given an output layer! Make sure the neural network is set-up with all layers before starting training\033[0m')
            return

        # number of mini-batches
        if sgd:
            minibatch_count = int(inputs.shape[0] / minibatch_size)

        # learning_schedule will be either a constant or decay from initial_learning_rate to final_learning_rate over the course of the epochs
        learning_schedule = lambda epoch: initial_learning_rate
        if final_learning_rate is not None and final_learning_rate != initial_learning_rate:
            t0 = initial_learning_rate * final_learning_rate / (initial_learning_rate - final_learning_rate) * epochs
            t1 = final_learning_rate / (initial_learning_rate - final_learning_rate) * epochs
            learning_schedule = lambda epoch: t0 / (t1 + epoch)

        # go over epochs
        for epoch in range(1, epochs + 1):

            # Eta will either always be the same, or go from initial_ to final_learning_rate over the epochs
            eta = learning_schedule(epoch-1)
            
            # Permute data each epoch
            perm = self.rng.permuted(np.arange(0, inputs.shape[0]))
            inputs = inputs[perm, :]
            targets = targets[perm, :]

            # Go through all minibatches in the input set
            if sgd:
                for m in range(minibatch_count):
                    idx = minibatch_size * int(self.rng.random() * minibatch_count)
                    ins = inputs[idx : idx + minibatch_size]
                    targs = targets[idx : idx + minibatch_size]
                    
                    self.back_prop(ins, targs, learning_rate=eta, regularization=regularization)
            else:
                self.back_prop(inputs, targets, learning_rate=eta, regularization=regularization)
            
            if verbose:
                print(f"[ Epoch: {epoch}/{epochs}; " + self.cost_function.error_name() + f": {self.cost_function.error_nn(self.feed_forward(inputs), targets)} ]")
                if testing_inputs is not None and testing_targets is not None:
                    print(f"\t\tTesting " + self.cost_function.error_name() + f": {self.error(testing_inputs, testing_targets)}")

        print()
        train_error = self.error(inputs, targets)
        print(f"[ Finished training with " + self.cost_function.error_name() + f": {train_error} ]")
        if testing_inputs is not None and testing_targets is not None:
            test_error = self.error(testing_inputs, testing_targets)
            print(f"\t\tTesting " + self.cost_function.error_name() + f": {test_error}")
            return train_error, test_error
        return train_error
    

    def grid_train(self, train_inputs: np.matrix, train_targets: np.matrix, test_inputs: np.matrix, test_targets: np.matrix, filename: str = None, plot: bool = True, sgd: bool = True, initial_learning_rate: float = 0.1, final_learning_rate: float = None, epochs: int = 1000, minibatch_size: int = 5, regularization: float = 0, reset_rng: bool = True, verbose: bool = False):
        """
            Grid searches amongst 2 parameters by repeatedly training and resetting the network
            Parameters:
                train_inputs (np.matrix): Training input data
                train_targets (np.matrix): Training output data
                test_inputs (np.matrix): Testing input data
                test_targets (np.matrix): Testing output data
                filename (str | None): Filename to save results to; if passing None, will not write results out at all
                plot (bool): Whether to plot the errors/accuracy scores in a contour plot
                sgd (bool): Whether to use stochastic gradient descent or gradient descent
                initial_learning_rate (float | list): If passing as a list, will create a grid search around the parameter
                final_learning_rate (float | list): If passing as a list, will create a grid search around the parameter
                epochs (int | list): If passing as a list, will create a grid search around the parameter
                minibatch_size (int | list): If passing as a list, will create a grid search around the parameter
                regularization (float | list): If passing as a list, will create a grid search around the parameter
                reset_rng (bool): Whether to reset the random number generator with the same seed between each iteration; if True, the same weights & biases will be used every time
                verbose (bool): Whether to print information about training as it occurs
        """

        # List all parameters that are ranges
        range_params = []
        const_params = []
        add_p = lambda name, val: range_params.append({'name':name, 'range':val}) if (isinstance(val, list) or isinstance(val, np.ndarray)) else const_params.append({'name':name, 'value':val})
        add_p('initial_learning_rate', initial_learning_rate)
        add_p('final_learning_rate', final_learning_rate)
        add_p('epochs', epochs)
        add_p('minibatch_size', minibatch_size)
        add_p('regularization', regularization)

        if len(range_params) > 2:
            print('\033[91mGrid training with more than 2 range parameters isn\'t supported right now! Please pass at most 2 range parameters.\033[0m')
            return
        if len(range_params) < 2:
            print('\033[91mGrid training with less than 2 range parameters isn\'t supported right now! Please pass at least 2 range parameters.\033[0m')
            return

        # Extract which parameters should be ranges
        param1 = range_params[0]['name']
        param2 = range_params[1]['name']
        param1_range = range_params[0]['range']
        param2_range = range_params[1]['range']
        params = dict()
        for param in const_params:
            params[param['name']] = param['value']

        # Run through grid search
        results = []
        results_mat = np.zeros((len(param1_range), len(param2_range)))
        for i, param1_val in enumerate(param1_range):
            for j, param2_val in enumerate(param2_range):

                # Set up parameters to use for this iteration
                params[param1] = param1_val
                params[param2] = param2_val

                # Reset and train
                self.reset(reset_rng=reset_rng)
                train_result, test_result = self.train(train_inputs, train_targets, verbose=verbose, testing_inputs=test_inputs, testing_targets=test_targets, sgd=sgd, **params)

                # Save results
                results_mat[i, j] = test_result
                results.append({
                    param1: param1_val,
                    param2: param2_val,
                    'train_err': train_result,
                    'test_err': test_result
                })

        # Save results to file
        if filename is not None:
            with open('results/' + filename + '.pickle', 'wb') as handle:
                d = {'date': time(), 'seed': self.random_state}
                for param in const_params:
                    d[param['name']] = param['value']
                d['results'] = results
                pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Plot results
        if plot:
            plt.figure()
            plt.contourf(param1_range, param2_range, results_mat)
            plt.xlabel(param1)
            plt.ylabel(param2)
            plt.colorbar()
            plt.show()
