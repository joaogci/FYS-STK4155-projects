
import numpy as np
from Splitter import Splitter
from sklearn.model_selection import train_test_split
from InputSet import InputSet
from PredictionSource import PredictionSource

class TrainTestSplitterSk(Splitter):
    """
        Splitter that splits the data into a training set and a (typically smaller) testing set
        Uses SciKitLearn implementation
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
            Splits the data into a training set and a test set
            Uses the SciKit Learn implementation (see https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self._test_size, random_state=self._seed)
        
        return {
            'full': InputSet(name='Full', X=X, y=y),
            'train': InputSet(name='Training set', X=X_train, y=y_train),
            'test': InputSet(name='Testing set', X=X_test, y=y_test)
        }

    def prediction_sources(self) -> list:
        # Use the training set only
        return [PredictionSource(
            name='Train/Test',
            src_set='train',
            dst_sets=['full', 'train', 'test']
        )]
