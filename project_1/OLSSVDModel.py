
import numpy as np
from Model import Model

class OLSSVDModel(Model):
    """
        Implementation of Ordinary Least Squares from SVD as a Model child class, to be given to a Solver instance
    """

    NAME = 'Ordinary Least Squares via SVD'

    def interpolate(self, design_matrix: np.matrix, y: np.matrix, degree: float) -> np.matrix:
        """
            Given a design matrix and a (training) data set, returns an evaluator function object that can be given additional data to make predictions
            Predictions will be based off OLS with SVD for this model
        """

        # Compute SVD of design matrix
        u, sigma, vT = np.linalg.svd(design_matrix, full_matrices=False)
        
        # Compute beta (slower, but allows to be then used to predict more than just data sets of the same length)
        beta = np.multiply(vT.T, 1 / sigma) @ u.T @ y

        return beta
