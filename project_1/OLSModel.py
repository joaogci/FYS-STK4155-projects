
import numpy as np
from Model import Model

class OLSModel(Model):
    """
        Implementation of the Ordinary Least Squares algorithm as a Model child class, to be given to a Solver instance
    """

    name = 'Ordinary Least Squares'

    def __init__(self, pseudo_inverse: bool = True):
        """
            Initializes the OLSModel
            Parameters:
                pseudo_inverse (bool): Whether to use SVD-based pseudo-inverse in computation of beta
        """
        self._pseudo_inverse = pseudo_inverse
        if self._pseudo_inverse:
            self.name = 'Ordinary Least Squares (SVD)'

    def interpolate(self, design_matrix: np.matrix, y: np.matrix) -> tuple:
        """
            Given a design matrix and a (training) data set, returns an evaluator function object that can be given additional data to make predictions
            Predictions will be based off OLS for this model
        """

        # Compute beta from the matrix inverse or pseudo inverse
        if self._pseudo_inverse: # Use SVD pseudo-inverse
            
            # Compute SVD of design matrix
            u, sigma, vT = np.linalg.svd(design_matrix, full_matrices=False)
            
            # Compute beta (slower, but allows to be then used to predict more than just data sets of the same length)
            beta = np.multiply(vT.T, 1 / sigma) @ u.T @ y
            
        else: # Use true matrix inverse (may be ill-conditioned)
            beta = np.linalg.inv(design_matrix.T @ design_matrix) @ design_matrix.T @ y

        return beta
