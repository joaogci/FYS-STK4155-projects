
import numpy as np
from Model import Model

class Ridge(Model):
    """
        Implementation of the Ridge regression as a Model child class, to be given for a Solver instance
    """
    
    NAME = "Ridge"
    
    def __init__(self, lmd: float, pseudo_inverse: bool = True):
        """
            Initializes the Ridge Model
            Parameters:
                lmd (float): Hyper-parameter for the Ridge regression
                pseudo_inverse (bool): Whether to use pseudo-inverse in computation of beta
        """
        self._lmd = lmd
        self._pseudo_inverse = pseudo_inverse
        
    def interpolate(self, design_matrix: np.matrix, y: np.matrix, degree: float) -> tuple:
        """
            Given a design matrix and a (training) data set, returns an evaluator function object that can be given additional data to make predictions
            Predictions will be based off Ridge for this model
        """

        # Compute beta with the peseudo inverse
        if self._pseudo_inverse: # Use numpy.linalg.pinv (uses SVD)
            beta = np.linalg.pinv(design_matrix.T @ design_matrix + self._lmd * np.eye(degree + 1)) @ design_matrix.T @ y
        else: # Use true matrix inverse (may be ill-conditioned)
            beta = np.linalg.inv(design_matrix.T @ design_matrix + self._lmd + np.eye(degree + 1)) @ design_matrix.T @ y

        # Curry over a prediction function to predict results from any design matrix (not just the one given to interpolate)
        def predict(X: np.matrix) -> np.matrix:
            prediction = X @ beta
            return prediction
        return predict, np.var(beta)


