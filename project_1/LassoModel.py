
import numpy as np
from sklearn.linear_model import Lasso
from Model import Model

class Ridge(Model):
    """
        Implementation of the Lasso regression as a Model child class, to be given for a Solver instance
    """
    
    NAME = "Ridge"
    
    def __init__(self, lmd: float):
        """
            Initializes the Lasso Model
            Parameters:
                lmd (float): Hyper-parameter for the Ridge regression
        """
        self._lmd = lmd
        
    def interpolate(self, design_matrix: np.matrix, y: np.matrix, degree: float) -> tuple:
        """
            Given a design matrix and a (training) data set, returns an evaluator function object that can be given additional data to make predictions
            Predictions will be based off Lasso for this model
        """
        
        # Uses sklearn function to compute the linear fit for the Lasso model

        lasso_train = Lasso(alpha=self._lmd)
        lasso_train.fit(design_matrix, y)
        
        beta = lasso_train.coef_

        # Curry over a prediction function to predict results from any design matrix (not just the one given to interpolate)
        def predict(X: np.matrix) -> np.matrix:
            prediction = X @ beta
            return prediction
        return predict, np.var(beta)


