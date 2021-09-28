
from PostProcess import PostProcess
from ErrorEstimates import r2, mse, beta_conf_intervals
import numpy as np

class ErrDisplayPostProcess(PostProcess):
    """
        Post process that prints the error estimates for the predictions made
    """

    def __init__(self, display_beta_conf_interval: bool = True, display_r2: bool = True, display_mse: bool = True):
        """
            Default constructor, allows setting whether to display R2/MSE/beta conf intervals
        """
        self._display_r2 = display_r2
        self._display_mse = display_mse
        self._display_beta_conf_interval = display_beta_conf_interval

    def run(self, data: tuple, sets: dict, prediction_sources: list, models: list, degree: int):
        """
            Prints the MSE and R2 score for the predictions made
        """
        print('\n#####\n')

        # Print on a model basis
        for model in models:
            print('--- ', model.name, ' ---\n')
            
            # Go through the different sources and print the MSE/R2/... for the different destinations of each source
            for prediction_src in prediction_sources:
                print('\t- Predictions made for', prediction_src.name)

                # Display the confidence intervals for betas predicted via this model + source
                if self._display_beta_conf_interval:
                    X = sets[prediction_src.src_set].get_src_design_mat()
                    print('\t  Beta confidence interval:', beta_conf_intervals(X))

                # Go through the prediction sets attached to the source set to display MSE/R2
                for dst in prediction_src.dst_sets:
                    dst_set = sets[dst]
                    print('\t\t-', dst_set.name)

                    y = dst_set.get_src_y()
                    pred = dst_set.get_prediction(model.name)

                    if self._display_mse:
                        _mse = mse(y, pred)
                        print('\t\t\tMean Squared Error:', _mse)
                    
                    if self._display_r2:
                        _r2 = r2(y, pred)
                        print('\t\t\tR2 Score:', _r2)
                    
                print()
            print()

        print('#####\n')
