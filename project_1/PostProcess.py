
import numpy as np
import abc

class PostProcess:
    """
        A component that can be added to a Solver instance to run code after predictions have been made, for example to plot the results
    """

    @abc.abstractmethod
    def run(self, data: tuple, design_matrices: dict, sets: dict, predictions: dict, betas: dict, degree: int):
        """
            Runs the post-process given the original data fed to the model and the prediction achieved
            Parameters:
                name (str): Name of the model that has been executed
                data (tuple): 2- or 3-component tuple containing the original data
                design_matrices (dict<str, np.matrix>): Split up design matrices (contains at least a 'full' key)
                sets (dict<str, np.matrix>): Split up data sets (contains at least a 'full' key set to the same as data[-1])
                predictions (dict<str, dict<str, np.matrix>>): Prediction matrices for each of the models, for each of the labeled sets
                betas (dict<str, np.matrix>): The feature matrices beta obtained from the full/training data for each model
                degree (int): The degree of the polynomial being predicted
        """
        print('Error: cannot instantiate/use the default PostProcess class - use a base class that overrides run()!')
