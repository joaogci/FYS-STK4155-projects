
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

    def run(self, data: tuple, sets: dict, prediction_sources: list, models: list, degree: int):
        """
            Displays a plot of the original and predicted data
        """

        title = self.title if self.title is not None else 'Predictions'

        # 2D
        if len(data) <= 2:

            for prediction_src in prediction_sources:

                plt.figure(title + ' ' + prediction_src.name)

                src = sets[prediction_src.src_set]
                main_dst = sets[prediction_src.dst_sets[-1]] # Choose the last of the destination sets as the 'main' destination, i.e. the testing set

                # Either display the entire input data, or split it up into the training and testing sets
                src_X = src.get_src_design_mat()
                dst_X = main_dst.get_src_design_mat()
                if src.name == main_dst.name:
                    plt.plot(src_X[:,1], src.get_src_y(), 'k+', label='Input data')
                else:
                    plt.plot(src_X[:,1], src.get_src_y(), 'k+', label='Input data (' + src.name + ')', alpha=0.25)
                    plt.plot(dst_X[:,1], main_dst.get_src_y(), 'k+', label='Input data (' + main_dst.name + ')')
                M = max(np.max(src_X[:,1]), np.max(dst_X[:,1]))
                m = min(np.min(src_X[:,1]), np.min(dst_X[:,1]))
                    
                # Display a smooth curve of the polynomial regardless of the input data, for each model
                x_display = np.linspace(m, M, self._display_steps)

                for model in models:
                    y_display = np.zeros(self._display_steps)
                    beta = src.get_beta(model.name)
                    for i in range(len(beta)):
                        y_display += beta[i] * x_display ** i

                    plt.plot(x_display, y_display, '--', label='Prediction (' + model.name + ')')
                
                plt.title(title + ' ' + prediction_src.name)
                plt.xlabel('x')
                plt.ylabel('y')
                plt.legend()

            if self.show:
                plt.show()
        
        # 3D
        else:

            # Compute partition of figures into a grid n x k with k <= 3
            num_plots = 1 + len(models) * len(prediction_sources)
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
            fig = plt.figure(title, figsize=(16, 10), dpi=80)
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
            for pred_src in prediction_sources:
                main_dst = sets[pred_src.dst_sets[-1]]
                for model in models:
                    pltIdx += 1
                    
                    # Show prediction as a smooth plot
                    ax = fig.add_subplot(num_lines, plot_partition[0], pltIdx, projection='3d')
                    
                    # Generate linspaced meshgrid to show predictions at smooth points
                    xm_display, ym_display = np.meshgrid(np.linspace(np.min(data[0]), np.max(data[0]), self._display_steps), np.linspace(np.min(data[1]), np.max(data[1]), self._display_steps))
                    zm_display = np.zeros((self._display_steps, self._display_steps))
                    betaIdx = 0
                    for i in range(degree + 1):
                        for k in range(i + 1):
                            zm_display += main_dst.get_beta(model.name)[betaIdx] * (xm_display ** (i - k)) * (ym_display ** k)
                            betaIdx += 1
                    surf = ax.plot_surface(xm_display, ym_display, zm_display, cmap=cm.coolwarm, linewidth=0, antialiased=True)
                    
                    ax.set_zlim(np.min(zm_display) - 0.3, np.max(zm_display) + 0.3)
                    ax.zaxis.set_major_locator(LinearLocator(10))
                    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

                    fig.colorbar(surf, shrink=0.5, aspect=5)
                    plt.title(model.name + ' prediction, ' + pred_src.name)
                    plt.xlabel('x')
                    plt.ylabel('y')

            if self.show:
                plt.show()
