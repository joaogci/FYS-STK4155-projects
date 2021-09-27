
import numpy as np
import abc

class PostProcess:
    """
        A component that can be added to a Solver instance to run code after predictions have been made, for example to plot the results
    """

    @abc.abstractmethod
    def run(self, data: tuple, sets: dict, prediction_sources: list, models: list, degree: int):
        """
            Runs the post-process given the original data fed to the model and the prediction achieved
            Parameters:
                name (str): Name of the model that has been executed
                data (tuple): 2- or 3-component tuple containing the original data
                sets (dict<str, InputSet>): The different sets the data has been split into
                prediction_sources (list<PredictionSource>): The mapping of sets used and
                                        which sets to use to create predictions for which
                models (list<Model>): The list of models used to predict the data
                degree (int): The degree of the polynomial being predicted
        """
        print('Error: cannot instantiate/use the default PostProcess class - use a base class that overrides run()!')
