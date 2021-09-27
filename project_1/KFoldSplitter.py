
from PredictionSource import PredictionSource
import numpy as np
from Splitter import Splitter
from InputSet import InputSet

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

    def _ordinal(self, k: int) -> str:
        """
            From the index of a fold, returns its ordinal designator
            0 -> "1st", 1 -> "2nd", 2 -> "3rd", etc.
        """
        suffix = ['th', 'st', 'nd', 'rd', 'th'][min((k + 1) % 10, 4)]
        return str(k + 1) + (suffix if k < 10 or k > 20 else 'th')

    def split(self, x: np.matrix, y: np.matrix) -> tuple:
        """
            Splits the design matrix and data into k sets
        """
        # Check inputs
        assert x.shape[0] == y.shape[0], "SizeError: KFold was given inputs of different sizes! Expects n_row(X) == len(y), given n_row(X) = {}, len(y) = {}!".format(x.shape[0], y.shape[0])
        
        # If there is a need to shuffle the data
        if self._shuffle:
            rng = np.random.default_rng(seed=self._seed)
            
            perm = rng.permuted(np.arange(0, x.shape[0]))
            x = x[perm]
            y = y[perm]
        
        # Output dictionary
        kfold = { }

        # Split into k folds
        KFold_size = np.floor(y.shape[0] / self._k)
        for k in range(self._k):
            train_idx = np.concatenate((np.arange(0, k * KFold_size, dtype=int), np.arange((k+1) * KFold_size, self._k * KFold_size, dtype=int)))
            test_idx = np.arange(k * KFold_size, (k+1) * KFold_size, dtype=int)
            
            kfold['full_' + str(k)] = InputSet(name='Full', X=x, y=y) # One version of the full set per fold, to make separate predictions of
            kfold['train_' + str(k)] = InputSet(name=self._ordinal(k) + ' training fold', X=x[train_idx], y=y[train_idx])
            kfold['test_' + str(k)] = InputSet(name=self._ordinal(k) + ' testing fold', X=x[test_idx], y=y[test_idx])

        return kfold
    
    def prediction_sources(self) -> list:
        # Use the different training sets as prediction sources
        sources = list()
        for k in range(self._k):
            # use each training set as the source for the corresponding training + testing set
            sources.append(PredictionSource(
                name=self._ordinal(k) + ' fold',
                src_set='train_' + str(k),
                dst_sets=['full_' + str(k), 'train_' + str(k), 'test_' + str(k)]
            ))
        return sources
