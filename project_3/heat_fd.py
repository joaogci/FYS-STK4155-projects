import numpy as np
import matplotlib.pyplot as plt

import sys

def finite_difference(nx, nt):
    """
        Solves u_t = u_xx by finite differences method.
        Stability condition:
            dt/dx**2 < 1/2
        Assuming:
            u(0, t) = u(1, t) = 0
            u(x, 0) = sin(pi * x)
    """
    x, t = np.linspace(0, 1, nx), np.linspace(0, 1, nt)
    dx, dt = x[1] - x[0], t[1] - t[0]
    
    if dt / dx**2 >= 1 / 2:
        print(f"dt / dx**2 = {dt / dx**2}.")
        print("Stability condition not achieved!; Change dx or dt")
        exit(1)
    
    print(f"Starting simulation with nx: {nx} and nt: {nt}.")
    
    u = np.zeros((nx, nt))
    u[:, 0] = np.sin(np.pi * x)
    
    for i in range(nt - 1):
        for j in range(2, nx - 1):
            u[j, i + 1] = u[j, i] + dt / dx**2 * (u[j - 1, i] - 2 * u[j, i] + u[j + 1, i])
        print(f" time iter: {i+1}/{nt}", end='\r')
    print()
    
    X, T = np.meshgrid(x, t)
    return X, T, u.T
    
if __name__ == '__main__':
    
    nx = int(sys.argv[1])
    nt = int(sys.argv[2])
    
    X, T, u_fd = finite_difference(nx, nt)
    print(u_fd.shape)

    g = lambda x, t: np.exp(-t * np.pi**2) * np.sin(np.pi*x)
    u_anal = g(X, T)
    
    abs_relativ_error = np.zeros_like(u_anal)
    abs_relativ_error = np.abs(u_anal - u_fd)
    
    print(f"Max error: {np.max(abs_relativ_error)}.")
    
    plt.figure("Finite Difference")
    plt.contourf(X, T, u_fd)
    plt.xlabel("$x$")
    plt.ylabel("$t$")
    plt.title("Solution to the heat equation using Finite Difference")
    plt.colorbar()
    
    plt.figure("Analytical Solution")
    plt.contourf(X, T, u_anal)
    plt.xlabel("$x$")
    plt.ylabel("$t$")
    plt.title("Analytical solution to the heat equation")
    plt.colorbar()
    
    plt.figure("Absolute Relative Error")
    plt.contourf(X, T, abs_relativ_error)
    plt.xlabel("$x$")
    plt.ylabel("$t$")
    plt.title("Absolute relative error of the NN compared with analytical solution")
    plt.colorbar()
    
    plt.show()
    