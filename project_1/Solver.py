
import numpy as np
from DataGenerator import DataGenerator
from Splitter import Splitter
from Model import Model
from PostProcess import PostProcess

class Solver:
    """
        Component-based solver implementation
        Components can be added to a solver instance to set up a model, a data generator, a scaler, etc.
    """


    def __init__(self, degree: int, data_generator: DataGenerator = None, splitter: Splitter = None, model: Model = None, post_processes = []):
        """
            Default Solver constructor
            Parameters can be used to set up components on the Solver in a non-verbose way
        """
        self._degree = degree
        self._data_generator = data_generator
        self._splitter = splitter
        self._model = model
        self._post_processes = post_processes
    
    
    def set_data_generator(self, data_generator: DataGenerator):
        """
            Sets the data generator on the solver instance
            Parameters:
                data_generator (DataGenerator): The data generator to use
        """
        self._data_generator = data_generator
    
    def set_splitter(self, splitter: Splitter):
        """
            Sets the splitter on the solver instance
            Parameters:
                splitter (Splitter): The splitter to use
        """
        self._splitter = splitter
    
    def set_model(self, model: Model):
        """
            Sets the model on the solver instance
            Parameters:
                model (Model): The model to use to predict the data
        """
        self._model = model
    
    def add_post_process(self, post_process: PostProcess):
        """
            Adds a post-process pass to the solver instance
            Parameters:
                post_process (PostProcess): The post process pass to append and run once predictions are made
        """
        self._post_processes.append(post_process)


    def _design_matrix(self, x1: np.matrix, x2: np.matrix) -> np.matrix:
        """
            Create the design matrix in the form of a Vandermonde matrix for one or two
            dimensional data set. The matrix is of the form

            [[1 x_1 x_1^2 ... x_1^(degree)]
             ...
             [1 x_n x_n^2 ... x_n^(degree)]]

            in 1D, and

            [[1 x_1 y_1 x_1^2 x_1y_1 y_1^2 ... y_1^(degree)]
             ...
             [1 x_n y_n x_n^2 x_ny_n y_n^2 ... y_n^(degree)]]

            in 2D.
            
            Parameters: 
                x1 (numpy array): The input data points on the X axis
                x2 (numpy array): The input data points on the Y axis, or None if generating 1D set
            
            Returns: 
                (numpy array): (n x p) dimensional matrix, 
                where n is number of datapoints and p is the degree 
                plus 1 (1-variable input) or p = degree*(degree + 1)/2 (2-variable input)
        """

        if x2 is None: # 1-variable Vandermonde matrix
            X = np.ones((len(x1), self._degree + 1)) # First column of design matrix is 1

            for i in range(1, self._degree+1): # First column is 1, so we skip it
                X[:, i] = x1.T ** i
        
        else: # 2-variable Vandermonde matrix
            X = np.ones((len(x1), int((self._degree + 1) * (self._degree + 2) / 2))) # First column of design matrix is 1

            for i in range(1, self._degree + 1): # First column is 1, so we skip it
                q = int(i * (i + 1) / 2) # 1 + 2 + ... + i
                for k in range(i + 1):
                    X[:, q + k] = (x1 ** (i - k)) * (x2 ** k)
        
        # Return design matrix
        return X



    def run(self):
        """
            Runs the data generator, model, and other attached components to create necessary prediction(s)
        """
        
        # Check setup
        if self._data_generator == None:
            print('Error: no data generator has been defined on the solver!')
            return
        if self._model == None:
            print('Error: no model has been defined on the solver!')
            return

        # Generate data
        data = self._data_generator.generate()
        
        # Create design matrix
        X_full = self._design_matrix(data[0], data[1] if len(data) > 2 else None)

        # Split data
        if self._splitter != None:
            X_split, y_split = self._splitter.split(X_full, data[-1])
        else:
            X_split = { 'full': X_full }
            y_split = { 'full': data[-1] }
        
        # Scale data
        # @todo
        
        # Init model and get evaluator to make predictions out of
        # Selecting which set to use out of the full set depending on the labeled sets in X_split and y_split
        X = X_split['full']
        y = y_split['full']
        if 'train' in X_split.keys(): # Use training data
            X = X_split['train']
            y = y_split['train']
        evaluator = self._model.interpolate(X, y)

        # Make predictions for all subsets
        predictions = {}
        for key in X_split.keys():
            predictions[key] = evaluator(X_split[key])

        # Run post-processes on original data + full prediction
        for process in self._post_processes:
            process.run(self._model.NAME, data, y_split, predictions)
