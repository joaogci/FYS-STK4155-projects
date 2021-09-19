
import numpy as np
from Splitter import Splitter
from sklearn.model_selection import train_test_split

class TrainTestSplitter(Splitter):
    """
        Splitter that splits the data into a training set and a (typically smaller) testing set
    """

    def __init__(self, test_size: float = 0.25):
        """
            Initializes the splitter, with a given test set size as a fraction of the total set
            Parameters:
                test_size (float): The test set size as a fraction of the total set
        """
        self._test_size = test_size

    
    def split(self, X: np.matrix, y: np.matrix) -> tuple:
        """
            Splits the data into a training set and a test set
            Uses the SciKit Learn implementation (see https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = self._test_size)
        return {
            'full': X,
            'train': X_train,
            'test': X_test
        }, {
            'full': y,
            'train': y_train,
            'test': y_test
        }
