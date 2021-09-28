
import numpy as np
from dataclasses import dataclass

@dataclass
class InputSet:
    """
        A named set of data used as an input for a Solver instance
    """

    name: str # Name of the set
    
    X: np.matrix # Design matrix
    y: np.matrix # Outcome data

    X_scaled: np.matrix = None # Scaled design matrix
    y_scaled: np.matrix = None # Scaled outcome data

    betas: dict = None  # Dictionary mapping { Model Name -> Beta matrix }; these betas are obtained from either the X or X_scaled data
                        # These beta values are not necessarily obtained from this set, but rather from the associated training set
    
    def get_src_design_mat(self) -> np.matrix:
        """
            Returns the design matrix to use as the source of the data
            Will be scaled if a scaled design matrix X_scaled has been defined on the set
            This is the matrix used when interpolating on the model
        """
        return self.X_scaled if self.X_scaled is not None else self.X
    
    def get_src_y(self) -> np.matrix:
        """
            Returns the data matrix to use as the source of the data
            Will be scaled if a scaled matrix y_scaled has been defined on the set
            This is the matrix used when interpolating on the model
        """
        return self.y_scaled if self.y_scaled is not None else self.y

    def set_beta(self, model_name: str, beta: np.matrix) -> None:
        """
            Sets the beta to use for this input set for a particular model
            The beta value may come from this set itself or from whatever set was defined as the source set
        """
        if self.betas is None:
            self.betas = dict()
        self.betas[model_name] = beta
    
    def get_beta(self, model_name: str) -> np.matrix:
        """
            Returns the beta associated with the specified model
        """
        if self.betas is None or model_name not in self.betas.keys():
            print('Error: Input Set', self.name, 'has not received a beta for model', model_name, '! Cannot return beta')
            return None
        return self.betas[model_name]
    
    def get_prediction(self, model_name: str) -> np.matrix:
        """
            Returns the prediction for this set for a particular model
        """
        if self.betas is None or model_name not in self.betas.keys():
            print('Error: Input Set', self.name, 'has not received a beta for model', model_name, '! Cannot make prediction')
            return None
        return self.get_src_design_mat() @ self.betas[model_name]
