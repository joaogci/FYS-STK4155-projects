
import numpy as np
from Model import Model

class OLSModel(Model):
    """
        Implementation of the Ordinary Least Squares algorithm as a Model child class, to be given to a Solver instance
    """

    NAME = 'Ordinary Least Squares'

    def __init__(self, pseudo_inverse: bool = True):
        """
            Initializes the OLSModel
            Parameters:
                pseudo_inverse (bool): Whether to use SVD-based pseudo-inverse in computation of beta
        """
        self._pseudo_inverse = pseudo_inverse

    def interpolate(self, design_matrix: np.matrix, y: np.matrix):
        """
            Given a design matrix and a (training) data set, returns an evaluator function object that can be given additional data to make predictions
            Predictions will be based off OLS for this model
        """

        # Compute beta from the matrix inverse or pseudo inverse
        if self._pseudo_inverse: # Use numpy.linalg.pinv (uses SVD)
            beta = np.linalg.pinv(design_matrix.T @ design_matrix) @ design_matrix.T @ y
        else: # Use true matrix inverse (may be ill-conditioned)
            beta = np.linalg.inv(design_matrix.T @ design_matrix) @ design_matrix.T @ y

        # Curry over a prediction function to predict results from any design matrix (not just the one given to interpolate)
        def predict(X: np.matrix) -> np.matrix:
            prediction = X @ beta
            return prediction
        return predict
