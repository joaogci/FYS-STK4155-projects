
from PostProcess import PostProcess
from ErrorEstimates import r2, mse
import numpy as np

class ErrDisplayPostProcess(PostProcess):
    """
        Post process that prints the error estimates for the predictions made
    """

    def __init__(self, display_r2: bool = True, display_mse: bool = True):
        """
            Default constructor, allows setting whether to display R2/MSE
        """
        self._display_r2 = display_r2
        self._display_mse = display_mse

    def run(self, name: str, data: tuple, prediction: np.matrix):
        """
            Prints the MSE and R2 score for the prediciton made
        """
        print('\n---')
        if self._display_mse:
            print('Mean Squared Error (' + name + '):', mse(data[-1], prediction))
        if self._display_r2:
            print('R2 Score (' + name + '):', r2(data[-1], prediction))
        print('---\n')
