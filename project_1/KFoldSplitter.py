
import numpy as np
from Splitter import Splitter

class KFoldSplitter(Splitter):
    """
        Splitter that splits the data into k sets, one for testing and k-1 for training
    """

    def __init__(self, k: int = 5, shuffle: bool = False, seed: int = 0):
        """
            Initializes the splitter, with a given test set size as a fraction of the total set
            Parameters:
                k (int): Number of subsets
        """
        self._k = k
        self._shuffle = shuffle
        self._seed = seed

    def split(self, X: np.matrix, y: np.matrix) -> tuple:
        """
            Splits the design matrix and data into k sets
        """
        # Check inputs
        assert X.shape[0] == y.shape[0], "SizeError: tt_split was given inputs of different sizes! Expects n_row(X) == len(y), given n_row(X) = {}, len(y) = {}!".format(X.shape[0], y.shape[0])
        
        X_KFold = dict()
        y_KFold = dict()
        
        # If there is a need to shuffle the data
        if self._shuffle:
            # Init random number generator
            rng = np.random.default_rng(seed=self._seed)
            
            perm = rng.permuted(np.arange(0, X.shape[0]))
            X = X[perm, :]
            y = y[perm]
        
        # Split into k folds
            
        return X_KFold, y_KFold
    