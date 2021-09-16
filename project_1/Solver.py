
import numpy as np
from DataGenerator import DataGenerator
from Model import Model
import matplotlib.pyplot as plt

class Solver:
    """
        Component-based solver implementation
        Components can be added to a solver instance to set up a model, a data generator, a scaler, etc.
    """


    def __init__(self, degree: int, data_generator: DataGenerator = None, model: Model = None):
        """
            Default Solver constructor
            Parameters can be used to set up components on the Solver in a non-verbose way
        """
        self._degree = degree
        self._data_generator = data_generator
        self._model = model
    
    
    def set_data_generator(self, data_generator: DataGenerator):
        """
            Sets the data generator on the solver instance
            Parameters:
                data_generator (DataGenerator): The data generator to use
        """
        self._data_generator = data_generator
    
    def set_model(self, model: Model):
        """
            Sets the model on the solver instance
            Parameters:
                model (Model): The model to use to predict the data
        """
        self._model = model


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

        if x2 == None: # 1-variable Vandermonde matrix
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
    

    def _r2(self, y_data: np.matrix, y_model: np.matrix) -> float:
        """
            Compute R2 score
            
            Parameters:
                y_data (vector) Input data points to compare against
                y_model (vector) Predicted data
                
            Returns:
                (float) The computed R2 score, which hopefully approaches 1
        """
        return 1 - np.sum(np.power(y_data - y_model, 2)) / np.sum(np.power(y_data - np.mean(y_data), 2))
    

    def _mse(self, y_data: np.matrix, y_model: np.matrix) -> float:
        """
            Compute Mean Squared Error
            
            Parameters:
                y_data (vector): Input data points to compare against
                y_model (vector): Predicted data
            
            Returns:
                (float) The computed Mean Squared Error, which hopefully approaches 0
        """
        return np.sum(np.power(y_data-y_model, 2)) / np.size(y_model)


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
        X = self._design_matrix(data[0], data[1] if len(data) > 2 else None)

        # Split/scale data
        # @todo
        
        # Init model and run evaluator on data set
        # @todo Run interpolate only on train data if applicable
        evaluator = self._model.interpolate(X, data[-1])
        prediction = evaluator(X, data[-1])

        # Show error estimates
        # @todo: implement as optional component
        print('Mean Squared Error:', self._mse(data[-1], prediction))
        print('R2 Score:', self._r2(data[-1], prediction))

        # Plot results
        # @todo: implement as optional component instead to have 2D/3D flexibility
        plt.plot(data[0], data[-1], 'k+', label='Input data')
        plt.plot(data[0], prediction, 'b-', label='Prediction')
        plt.legend()
        plt.show()
