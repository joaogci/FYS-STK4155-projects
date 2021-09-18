
import numpy as np
import abc

class PostProcess:
    """
        A component that can be added to a Solver instance to run code after predictions have been made, for example to plot the results
    """

    @abc.abstractmethod
    def run(self, name: str, data: tuple, prediction: np.matrix):
        """
            Runs the post-process given the original data fed to the model and the prediction achieved
            Parameters:
                name (str): Name of the model that has been executed
                data (tuple): 2- or 3-component tuple containing the original data
                prediction (np.matrix): Matrix containing the predicted results from the data
        """
        print('Error: cannot instantiate/use the default PostProcess class - use a base class that overrides run()!')
