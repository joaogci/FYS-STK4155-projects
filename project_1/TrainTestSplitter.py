
import numpy as np
from Splitter import Splitter

class TrainTestSplitter(Splitter):
    """
        Splitter that splits the data into a training set and a (typically smaller) testing set
    """

    def __init__(self, test_size: float = 0.25, seed: int = 0):
        """
            Initializes the splitter, with a given test set size as a fraction of the total set
            Parameters:
                test_size (float): The test set size as a fraction of the total set
        """
        self._test_size = test_size
        self._seed = seed

    def split(self, X: np.matrix, y: np.matrix) -> tuple:
        """
            Splits the design matrix and data into two sets of data; testing and training
        """
        # Check inputs
        assert X.shape[0] == y.shape[0], "SizeError: tt_split was given inputs of different sizes! Expects n_row(X) == len(y), given n_row(X) = {}, len(y) = {}!".format(X.shape[0], y.shape[0])
            
        # Init random number generator
        rng = np.random.default_rng(seed=self._seed)

        # Split the data into train and test sets
        split_size = int(X.shape[0] * (1 - self._test_size))
        
        perm = rng.permuted(np.arange(0, X.shape[0]))
        perm_X = X[perm, :]
        perm_y = y[perm]
        
        X_train = perm_X[0:split_size, :]
        y_train = perm_y[0:split_size]
        X_test  = perm_X[split_size:, :]
        y_test  = perm_y[split_size:]
            
        return {
            'full': X,
            'train': X_train,
            'test': X_test
        }, {
            'full': y,
            'train': y_train,
            'test': y_test
        }
