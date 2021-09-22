
import numpy as np
from sklearn.linear_model import Lasso
from Model import Model

class LassoModel(Model):
    """
        Implementation of the Lasso regression as a Model child class, to be given for a Solver instance
    """
    
    def __init__(self, lmd: float):
        """
            Initializes the Lasso Model
            Parameters:
                lmd (float): Hyper-parameter for the Ridge regression
        """
        self._lmd = lmd
        self.NAME = "Lasso lmd = " + str(self._lmd)
        
    def interpolate(self, design_matrix: np.matrix, y: np.matrix, degree: float) -> tuple:
        """
            Given a design matrix and a (training) data set, returns an evaluator function object that can be given additional data to make predictions
            Predictions will be based off Lasso for this model
        """
        
        # Uses sklearn function to compute the linear fit for the Lasso model

        lasso_train = Lasso(alpha=self._lmd, fit_intercept=False)
        lasso_train.fit(design_matrix, y)
        
        beta = lasso_train.coef_

        return beta


