
from PostProcess import PostProcess
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter

class PlotPostProcess(PostProcess):
    """
        Post process that displays a plot of the prediction
    """

    def run(self, name: str, data: tuple, sets: dict, predictions: dict, estimator_variance: float):
        """
            Displays a plot of the original and predicted data
            Note that only the full prediction is shown on the diagram
        """

        # 2D
        if len(data) <= 2:
            plt.figure(name + ' prediction')
            plt.plot(data[0], sets['full'], 'k+', label='Input data')
            plt.plot(data[0], predictions['full'], 'b-', label='Prediction')
            plt.title(name + ' prediction')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.legend()
            plt.show()
        
        # 3D
        else:
            # Reformat data from 1D to 2D matrices
            root = int(np.sqrt(len(data[0])))
            xm = np.zeros((root, root))
            ym = np.zeros((root, root))
            zm = np.zeros((root, root))
            zm_pred = np.zeros((root, root))
            for x in range(root):
                for y in range(root):
                    xm[x,y] = data[0][x * root + y]
                    ym[x,y] = data[1][x * root + y]
                    zm[x,y] = data[-1][x * root + y]
                    zm_pred[x,y] = predictions['full'][x * root + y]
            
            # Show input data
            fig = plt.figure('Input data', figsize=(8, 6), dpi=80)
            ax = fig.add_subplot(111, projection='3d')
            
            surf = ax.plot_surface(xm, ym, zm, cmap=cm.coolwarm, linewidth=0, antialiased=True)
            
            ax.set_zlim(np.min(data[-1]) - 0.3, np.max(data[-1]) + 0.3)
            ax.zaxis.set_major_locator(LinearLocator(10))
            ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

            fig.colorbar(surf, shrink=0.5, aspect=5)
            plt.title('Input data')
            plt.xlabel('x')
            plt.ylabel('y')

            # Show prediction
            fig = plt.figure(name + ' prediction', figsize=(8, 6), dpi=80)
            ax = fig.add_subplot(111, projection='3d')
            
            surf = ax.plot_surface(xm, ym, zm_pred, cmap=cm.coolwarm, linewidth=0, antialiased=True)
            
            ax.set_zlim(np.min(predictions['full']) - 0.3, np.max(predictions['full']) + 0.3)
            ax.zaxis.set_major_locator(LinearLocator(10))
            ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

            fig.colorbar(surf, shrink=0.5, aspect=5)
            plt.title(name + ' prediction')
            plt.xlabel('x')
            plt.ylabel('y')

            plt.show()
