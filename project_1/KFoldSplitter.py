
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

    def split(self, x: np.matrix, y: np.matrix) -> tuple:
        """
            Splits the design matrix and data into k sets
        """
        # Check inputs
        assert x.shape[0] == y.shape[0], "SizeError: KFold was given inputs of different sizes! Expects n_row(X) == len(y), given n_row(X) = {}, len(y) = {}!".format(x.shape[0], y.shape[0])
        
        # Output dictionaries
        x_KFold = dict()
        y_KFold = dict()
        
        # If there is a need to shuffle the data
        if self._shuffle:
            rng = np.random.default_rng(seed=self._seed)
            
            perm = rng.permuted(np.arange(0, x.shape[0]))
            x = x[perm]
            y = y[perm]
        
        # Split into k folds
        KFold_size = np.floor(y.shape[0] / self._k)
        
        for k in range(self._k):
            train_idx = np.concatenate((np.arange(0, k * KFold_size, dtype=int), np.arange((k+1) * KFold_size, self._k * KFold_size, dtype=int)))
            test_idx = np.arange(k * KFold_size, (k+1) * KFold_size, dtype=int)
            
            x_KFold['train_' + str(k)] = x[train_idx]
            x_KFold['test_' + str(k)] = x[test_idx]
            
            y_KFold['train_' + str(k)] = y[train_idx]
            y_KFold['test_' + str(k)] = y[test_idx]

        return x_KFold, y_KFold
    