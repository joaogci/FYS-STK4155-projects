
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm

def plot_prediction_3D(beta: np.matrix, degree: int, min_x: float = 0, max_x: float = 1, min_y: float = 0, max_y: float = 1, name: str = 'Prediction', display_steps: int = 100) -> None:
    """
        Given a beta feature matrix of a certain degree, plots the estimate of a 2D function over a certain domain

        Parameters:
            beta (np.matrix): Feature estimate given by the model
            degree (int): Polynomial degree
            min_x (float): Start of the domain to show on X
            max_x (float): End of the domain to show on X
            min_y (float): Start of the domain to show on Y
            max_y (float): End of the domain to show on Y
            name (str): Name of the plot to display
            display_steps (int): Number of points to show on X and Y; the overall number of data points computed is display_steps^2
    """

    # Show prediction as a smooth plot
    fig = plt.figure(name)
    ax = plt.axes(projection='3d')

    # Generate linspaced meshgrid to show predictions at smooth points
    xm_display, ym_display = np.meshgrid(np.linspace(min_x, max_x, display_steps), np.linspace(min_y, max_y, display_steps))
    zm_display = np.zeros((display_steps, display_steps))
    betaIdx = 0
    for i in range(degree + 1):
        for k in range(i + 1):
            zm_display += beta[betaIdx] * (xm_display ** (i - k)) * (ym_display ** k)
            betaIdx += 1
    surf = ax.plot_surface(xm_display, ym_display, zm_display, cmap=cm.gray, linewidth=0, antialiased=True)

    # Plot surface
    ax.set_zlim(np.min(zm_display) - 1, np.max(zm_display) + 1)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(name)
    plt.show()
