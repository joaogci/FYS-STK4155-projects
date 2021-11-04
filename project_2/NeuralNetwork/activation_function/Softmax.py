
import numpy as np
from .ActivationFunction import ActivationFunction

class Softmax(ActivationFunction):
    """
        Softmax activation function
        Note: since Softmax needs ALL of the layer's weighted sums at once instead of just a single float, the inputs to call and d MUST be matrices
    """

    def __call__(self, x: np.matrix) -> np.matrix:
        """
            Returns f(x)
        """
        # ln(sum(a_i)) = ln(a_0) + ln(sum(a_i / a_0))
        # ^ This might give better results without completely exploding, @todo try this
        expTerm = np.exp(x)
        return expTerm / np.sum(expTerm, axis=1, keepdims=True)

    def d(self, x: np.matrix) -> np.matrix:
        """
            Returns f'(x)
        """
        print('\n-Unimplemented softmax derivative-\n')
        return None
