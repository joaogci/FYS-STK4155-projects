
from PostProcess import PostProcess
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import math

class PlotPostProcess(PostProcess):
    """
        Post process that displays a plot of the prediction
    """

    def __init__(self, display_steps: int = 500, show: bool = True, title: str = None):
        self._display_steps = display_steps
        self.show = show
        self.title = title

    def run(self, data: tuple, design_matrices: dict, sets: dict, predictions: dict, betas: dict, degree: int):
        """
            Displays a plot of the original and predicted data
        """

        # 2D
        if len(data) <= 2:

            plt.figure(self.title if self.title is not None else 'Predictions')

            # Either display the entire input data, or split it up into the training and testing sets
            a = lambda model_name: 0
            
            if 'test_scaled' in design_matrices.keys():
                plt.plot(design_matrices['train_scaled'][:,1], sets['train_scaled'], 'k+', label='Input data (training set)', alpha=0.25)
                plt.plot(design_matrices['test_scaled'][:,1], sets['test_scaled'], 'k+', label='Input data (test set)')
                
                M = np.max(design_matrices['test_scaled'][:,1]) if np.max(design_matrices['test_scaled'][:,1]) > np.max(design_matrices['train_scaled'][:,1]) else np.max(design_matrices['train_scaled'][:,1])
                m = np.min(design_matrices['test_scaled'][:,1]) if np.min(design_matrices['test_scaled'][:,1]) < np.min(design_matrices['train_scaled'][:,1]) else np.min(design_matrices['train_scaled'][:,1])
                a = lambda model_name: np.mean(predictions[model_name]['test_scaled'])
            elif 'test' in design_matrices.keys():
                plt.plot(design_matrices['train'][:,1], sets['train'], 'k+', label='Input data (training set)', alpha=0.25)
                plt.plot(design_matrices['test'][:,1], sets['test'], 'k+', label='Input data (test set)')
                
                M = np.max(design_matrices['test'][:,1]) if np.max(design_matrices['test'][:,1]) > np.max(design_matrices['train'][:,1]) else np.max(design_matrices['train'][:,1])
                m = np.min(design_matrices['test'][:,1]) if np.min(design_matrices['test'][:,1]) < np.min(design_matrices['train'][:,1]) else np.min(design_matrices['train'][:,1])
            else:
                plt.plot(data[0], sets['full'], 'k+', label='Input data')
                
                M = np.max(data[0])
                m = np.min(data[0])
                
            # Display a smooth curve of the polynomial regardless of the input data, for each model
            x_display = np.linspace(m, M, self._display_steps)
            
            for model_name in betas.keys():
                print(a(model_name))
                y_display = np.zeros(self._display_steps) + a(model_name)
                for i in range(len(betas[model_name])):
                    y_display += betas[model_name][i] * x_display ** i

                plt.plot(x_display, y_display, '--', label='Prediction (' + model_name + ')')
            
            plt.title(self.title if self.title is not None else 'Predictions')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.legend()
            if self.show:
                plt.show()
        
        # 3D
        else:

            # Compute partition of figures into a grid n x k with k <= 3
            num_plots = 1 + len(betas)
            num_lines = int(math.ceil(num_plots / float(3)))
            plot_partition = [0] * num_lines
            for i in range(num_plots):
                plot_partition[i % num_lines] += 1

            # Reformat data from 1D to 2D matrices
            root = int(np.sqrt(len(data[0])))
            xm = np.zeros((root, root))
            ym = np.zeros((root, root))
            zm = np.zeros((root, root))
            for x in range(root):
                for y in range(root):
                    xm[x,y] = data[0][x * root + y]
                    ym[x,y] = data[1][x * root + y]
                    zm[x,y] = data[-1][x * root + y]
            
            # Show input data
            fig = plt.figure(self.title if self.title is not None else 'Predictions', figsize=(16, 10), dpi=80)
            ax = fig.add_subplot(num_lines, plot_partition[0], 1, projection='3d')
            
            surf = ax.plot_surface(xm, ym, zm, cmap=cm.coolwarm, linewidth=0, antialiased=True)
            
            ax.set_zlim(np.min(data[-1]) - 0.3, np.max(data[-1]) + 0.3)
            ax.zaxis.set_major_locator(LinearLocator(10))
            ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

            fig.colorbar(surf, shrink=0.5, aspect=5)
            plt.title('Input data')
            plt.xlabel('x')
            plt.ylabel('y')

            pltIdx = 1
            for model_name in betas.keys():
                pltIdx += 1
                
                # Show prediction as a smooth plot
                ax = fig.add_subplot(num_lines, plot_partition[0], pltIdx, projection='3d')
                
                # Generate linspaced meshgrid to show predictions at smooth points
                xm_display, ym_display = np.meshgrid(np.linspace(np.min(data[0]), np.max(data[0]), self._display_steps), np.linspace(np.min(data[1]), np.max(data[1]), self._display_steps))
                zm_display = np.zeros((self._display_steps, self._display_steps))
                betaIdx = 0
                for i in range(degree + 1):
                    for k in range(i + 1):
                        zm_display += betas[model_name][betaIdx] * (xm_display ** (i - k)) * (ym_display ** k)
                        betaIdx += 1
                surf = ax.plot_surface(xm_display, ym_display, zm_display, cmap=cm.coolwarm, linewidth=0, antialiased=True)
                
                ax.set_zlim(np.min(zm_display) - 0.3, np.max(zm_display) + 0.3)
                ax.zaxis.set_major_locator(LinearLocator(10))
                ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

                fig.colorbar(surf, shrink=0.5, aspect=5)
                plt.title(model_name + ' prediction')
                plt.xlabel('x')
                plt.ylabel('y')

            if self.show:
                plt.show()
