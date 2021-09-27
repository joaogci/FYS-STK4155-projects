
import numpy as np
from dataclasses import dataclass

@dataclass
class PredictionSource:
    """
        A named source of predictions that carries into a series of input sets
    """

    name: str # Name of the prediction source
    
    src_set: str # Name of the set to use as prediction source, on which the model will be trained
    dst_sets: list # Name of the sets onto which predictions will be applied
