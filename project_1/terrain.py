
import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm
import numpy as np
import math


# Terrain files included in res/
TERRAIN_1 = 'SRTM_data_Norway_1'
TERRAIN_2 = 'SRTM_data_Norway_2'


def load_terrain(name: str, min_xy: float = 0, max_xy: float = 1, min_z: float = 0, max_z: float = 1, rng: np.random.Generator = None, downsample: int = 1, scissor: float = None, sparse_sample: float = 0.5, plot: bool = False, show_plot: bool = True, save_fig: bool = False) -> np.matrix:
    """
        From an image filename, loads the terrain matrix with optional downsampling

        Parameters:
            name (str): Name of the tif file to load in
            min_xy (float|None): Desired minimum X & Y to use in order to scale the axes; if None, won't scale
            max_xy (float|None): Desired maximum X & Y to use in order to scale the axes; if None, won't scale
            min_z (float|None): Desired minimum Z to use in order to scale the input data; if None, won't scale
            max_z (float|None): Desired maximum Z to use in order to scale the input data; if None, won't scale
            rng (np.random.Generator|None): Random number generator to use to sample the input data non linearly (None to sample linearly)
            downsample (int): Optional value (>1) to downsample by nearest filtering - only every nth value will be kept
            scissor (float|None): If !None, the fraction of the normal terrain to keep (cropping out the rest)
            sparse_sample (float): If rng is not None, the fraction of data points to keep out of the initial data
            plot (bool): Whether to show a 3D plot of the input data before returning the results
            show_plot (bool): Whether to call matplotlib.pyplot.show() after generating the plot; unused if plot == False

        Returns:
            (np.matrix): Numpy array of Xs
            (np.matrix): Numpy array of Ys
            (np.matrix): Numpy array of Zs/heights corresponding to (x,y) pairs
    """

    # Load full res image
    terrain = np.matrix(imread('res/' + name + '.tif'))

    # Select the biggest possible square shape
    N = min(terrain.shape[0], terrain.shape[1])
    terrain = terrain[:N, :N]

    # Crop it if necessary
    if scissor is not None:
        N = int(N*scissor)
        terrain = terrain[:N, :N]

    # Downsample if needed
    if downsample > 1:
        terrain = terrain[::downsample, ::downsample]
        N = terrain.shape[0] # i.e. N = N / downsample
    
    # Remap (normalize to 0..1 by default) input data (invlerp -> lerp)
    if min_z is not None and max_z is not None:
        terrain = (terrain - np.min(terrain)) / (np.max(terrain) - np.min(terrain)) * (max_z - min_z) + min_z
    
    if rng is not None:
        # Non-linearly distributed x & y
        x = np.sort(rng.uniform(0, N, math.floor(N * sparse_sample)))
        y = np.sort(rng.uniform(0, N, math.floor(N * sparse_sample)))
        x, y = np.meshgrid(x, y)
        z = np.zeros(x.shape)
        for u in range(z.shape[0]):
            for v in range(z.shape[1]):
                z[u, v] = terrain[int(x[u,v]), int(y[u,v])]
        terrain = z
    else:
        # Linearly distributed x & y
        x = np.linspace(0, N, N)
        y = np.linspace(0, N, N)
        x, y = np.meshgrid(x, y)
    
    # Remap (normalize to 0..1 by default) x and y (invlerp -> lerp)
    if min_xy is not None and max_xy is not None:
        x = (x - np.min(x)) / (np.max(x) - np.min(x)) * (max_xy - min_xy) + min_xy
        y = (y - np.min(y)) / (np.max(y) - np.min(y)) * (max_xy - min_xy) + min_xy

    # Optionally show a 3D plot of the terrain
    if plot:
        fig = plt.figure(name)
        ax = plt.axes(projection='3d')
        surf = ax.plot_surface(x, y, terrain, cmap=cm.gray, linewidth=0, antialiased=True)
        ax.set_zlim([np.min(terrain)-1, np.max(terrain)+1])
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.title(name)
        plt.xlabel('x')
        plt.ylabel('y')
        if show_plot:
            plt.show()
        if save_fig:
            plt.savefig("./images/ex6_original_data.pdf", dpi=400)

    ravel = lambda m: np.ravel(m).reshape((np.ravel(m).shape[0], 1))
    return ravel(x), ravel(y), ravel(terrain)


# If running the script directly, show a plot of the 1st terrain file
if __name__ == '__main__':
    load_terrain(TERRAIN_1, downsample=0.1, plot=True)
