import numpy as np
import matplotlib.pyplot as plt

n = 1000
dx = np.linspace(0, 20, n)
dt = np.linspace(0, 100, n)
dX, dT = np.meshgrid(dx, dt)

stability = dT / dX**2

region = np.zeros_like(stability)

for i in range(n):
    for j in range(n):
        if stability[i, j] < 1 / 2:
            region[i, j] = 1

plt.figure("Stability Region")
plt.title("Stability region for finite differences")
plt.contourf(dX, dT, region)
plt.xlabel("$\Delta x$")
plt.ylabel("$\Delta t$")
plt.colorbar()

plt.savefig("figs/stability_region.pdf")
