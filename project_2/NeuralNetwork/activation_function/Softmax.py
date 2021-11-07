
import numpy as np
from .ActivationFunction import ActivationFunction

# e^7; with a high precision
EXP_7 = 1096.6331584284585992637202382881214324422191348336131437827392407

class Softmax(ActivationFunction):
    """
        Softmax activation function
        Note: since Softmax needs ALL of the layer's weighted sums at once instead of just a single float, the inputs to call and d MUST be matrices
    """
    
    def name(self) -> str:
        return 'Softmax'

    def __call__(self, x: np.matrix) -> np.matrix:
        """
            Returns f(x)
        """
        # The following explodes very quickly, so we compute the logarithm of Softmax instead
        # expTerm = np.exp(x)
        # return expTerm / np.sum(expTerm, axis=1, keepdims=True)

        # ln(sum(a_i)) = n + ln(sum(a_i / e^n)) for any n; in particular we choose n=7 here since e^7 ~ 1100 which works nicely for most inputs
        ln_softmax = x - 7 - np.log(np.sum(np.exp(x)/EXP_7, axis=1))
        return np.exp(ln_softmax)

    def d(self, x: np.matrix) -> np.matrix:
        """
            Returns f'(x)
        """
        softmax_x = self(x)
        return np.multiply((1.0 - softmax_x), softmax_x)
