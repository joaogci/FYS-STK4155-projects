import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import sys

from NNSolver import DiffEqNet

class pde2(DiffEqNet):
    def __init__(self, layers, x, y, learning_rate=0.001):
        super(pde2, self).__init__(layers, learning_rate)

        self.x = x
        self.y = y
        self.var = tuple([x, y])

    @tf.function 
    def trial_func(self, x, y):
        return x**2 + y**2 + x + y + 1 + x * (x-1) * y * (y-1) * tf.squeeze(self(tf.stack([x, y], axis = 1), training = False))

    @tf.function
    def cost_function(self, x, y):
        with tf.GradientTape(persistent = True) as g:
            g.watch([x, y])
            with tf.GradientTape(persistent = True) as gg:
                gg.watch([x, y])
                trial = self.trial_func(x, y)

            dx_trial = gg.gradient(trial, x)
            dy_trial = gg.gradient(trial, y)

        dx2_trial = g.gradient(dx_trial, x)
        dy2_trial = g.gradient(dy_trial, y)

        del g
        del gg

        loss = (dx2_trial + dy2_trial - 4)**2

        return loss

layers = [2, 50, 10, 1]

nx = int(sys.argv[1])
nt = int(sys.argv[2])

npx = np.matrix(np.linspace(0, 1, nx)).reshape(-1, 1)
x = tf.cast(tf.convert_to_tensor(npx), tf.float32)

npt = np.matrix(np.linspace(0, 1, nt)).reshape(-1, 1)
t = tf.cast(tf.convert_to_tensor(npt), tf.float32)

Xtf, Ttf = tf.meshgrid(x, t)
X = tf.reshape(Xtf, [-1])
T = tf.reshape(Ttf, [-1])

u = pde2(layers, X, T, learning_rate=1e-3)

u.train(1000)

pred = tf.reshape(u.predict([X, T]), [nt, nx])

X, T = np.meshgrid(np.linspace(0, 1, nx), np.linspace(0, 1, nt))
def g(x, y):
    return x**2 + y**2 + x + y + 1
analytical = g(X, T)

abs_relativ_error = np.zeros_like(analytical)
abs_relativ_error = np.abs(analytical - pred)

print(f"Max error: {np.max(abs_relativ_error)}")

plt.figure("Prediction")
plt.contourf(X, T, pred)
plt.xlabel("$x$")
plt.ylabel("$t$")
plt.title("Solution to the heat equation with NN")
plt.colorbar()

plt.figure("Analytical")
plt.contourf(X, T, analytical)
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
