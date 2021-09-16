
import numpy as np
from Model import Model

class OLSSVDModel(Model):
    """
        Implementation of Ordinary Least Squares from SVD as a Model child class, to be given to a Solver instance
    """

    def __init__(self, compute_beta: bool = False):
        """
            Initializes the OLSSVDModel
            Parameters:
                compute_beta (bool): Whether to compute beta when running through the interpolation - slower if True
        """
        self._compute_beta = compute_beta

    def interpolate(self, design_matrix: np.matrix, y: np.matrix):
        """
            Given a design matrix and a (training) data set, returns an evaluator function object that can be given additional data to make predictions
            Predictions will be based off OLS with SVD for this model
        """

        # Compute SVD of design matrix
        u, sigma, vT = np.linalg.svd(design_matrix, full_matrices=False)
        
        # Compute beta optionally (slower)
        beta = None
        if self._compute_beta:
            beta = np.multiply(vT.T, 1 / sigma) @ u.T @ y

        # Curry over a prediction function to predict results from any data set (not just the one given to interpolate)
        uuT = u @ u.T
        def predict(X: np.matrix, y: np.matrix) -> np.matrix:
            prediction = uuT @ y
            return prediction
        return predict
