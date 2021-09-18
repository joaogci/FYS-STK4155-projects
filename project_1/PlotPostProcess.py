
from PostProcess import PostProcess
from ErrorEstimates import r2, mse
import numpy as np
import matplotlib.pyplot as plt

class PlotPostProcess(PostProcess):
    """
        Post process that displays a plot of the prediction
    """

    def run(self, name: str, data: tuple, prediction: np.matrix):
        """
            Displays a plot of the original and predicted data
        """

        # 2D
        plt.plot(data[0], data[-1], 'k+', label='Input data')
        plt.plot(data[0], prediction, 'b-', label='Prediction')
        plt.title(name + ' prediction')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.show()
