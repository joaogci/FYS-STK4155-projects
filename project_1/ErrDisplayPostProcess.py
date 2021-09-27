
from PostProcess import PostProcess
from ErrorEstimates import r2, mse, beta_conf_intervals
import numpy as np

class ErrDisplayPostProcess(PostProcess):
    """
        Post process that prints the error estimates for the predictions made
    """

    def __init__(self, display_beta_conf_interval: bool = True, display_r2: bool = True, display_mse: bool = True):
        """
            Default constructor, allows setting whether to display R2/MSE
        """
        self._display_r2 = display_r2
        self._display_mse = display_mse
        self._display_beta_conf_interval = display_beta_conf_interval

    def run(self, data: tuple, design_matrices: dict, sets: dict, predictions: dict, betas: dict, degree: int):
        """
            Prints the MSE and R2 score for the prediciton made
        """
        print('\n---\n')

        for model_name in predictions.keys():
            
            if 'train_scaled' in design_matrices.keys():
                if self._display_mse:
                    print('Mean Squared Error (' + model_name + ', scaled):', mse(sets['full_scaled'], predictions[model_name]['full_scaled']))
                    for key in sets.keys():
                        if key != 'full_scaled' and 'scaled' in key:
                            print('Mean Squared Error (' + model_name + ', ' + key.replace('_', ' ') + ' set):', mse(sets[key], predictions[model_name][key]))

                if self._display_mse and self._display_r2:
                    print()
                
                if self._display_r2:
                    print('R2 Score (' + model_name + ', scaled):', r2(sets['full_scaled'], predictions[model_name]['full_scaled']))
                    for key in sets.keys():
                        if key != 'full_scaled' and 'scaled' in key:
                            print('R2 Score (' + model_name + ', ' + key.replace('_', ' ') + ' set):', r2(sets[key], predictions[model_name][key]))
                
                if self._display_r2 and self._display_beta_conf_interval:
                    print()
                
                if self._display_beta_conf_interval: 
                    print('Beta confidence interval (' + model_name + ', scaled):\n', beta_conf_intervals(design_matrices['full_scaled']))
                    for key in sets.keys():
                        if key != 'full_scaled' and 'scaled' in key:
                            print('Beta confidence interval (' + model_name + ', ' + key.replace('_', ' ') + ' set):\n', beta_conf_intervals(design_matrices[key]))

            else:
                if self._display_mse:
                    print('Mean Squared Error (' + model_name + '):', mse(sets['full'], predictions[model_name]['full']))
                    for key in sets.keys():
                        if key != 'full':
                            print('Mean Squared Error (' + model_name + ', ' + key.replace('_', ' ') + ' set):', mse(sets[key], predictions[model_name][key]))

                if self._display_mse and self._display_r2:
                    print()
                
                if self._display_r2:
                    print('R2 Score (' + model_name + '):', r2(sets['full'], predictions[model_name]['full']))
                    for key in sets.keys():
                        if key != 'full':
                            print('R2 Score (' + model_name + ', ' + key.replace('_', ' ') + ' set):', r2(sets[key], predictions[model_name][key]))
                
                if self._display_r2 and self._display_beta_conf_interval:
                    print()
                
                if self._display_beta_conf_interval: 
                    print('Beta confidence interval (' + model_name + '):\n', beta_conf_intervals(design_matrices['full']))
                    for key in sets.keys():
                        if key != 'full':
                            print('Beta confidence interval (' + model_name + ', ' + key.replace('_', ' ') + ' set):\n', beta_conf_intervals(design_matrices[key]))

            print()

        print('---\n')
