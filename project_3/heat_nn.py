import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import sys

from NNSolver import DiffEqNet

class heateq(DiffEqNet):
    def __init__(self, layers, x, t, learning_rate=0.001):
        super(heateq, self).__init__(layers, learning_rate)

        self.x = x
        self.t = t
        self.var = tuple([x, t])

    @tf.function 
    def trial_func(self, x, t):
        # return x*(1-x)*t*tf.squeeze(self(tf.stack([x, t], axis = 1), training = False))
        # return tf.sin(np.pi*x) + t + x*(1-x)*t*tf.squeeze(self(tf.stack([x, t], axis = 1), training = False))
        return tf.sin(np.pi*x)*(1-t) + x*(1-x)*t*tf.squeeze(self(tf.stack([x, t], axis = 1), training = False))

    @tf.function
    def cost_function(self, x, t):
        with tf.GradientTape(persistent = True) as g:
            g.watch([x, t])
            with tf.GradientTape(persistent = True) as gg:
                gg.watch([x, t])
                trial = self.trial_func(x, t)

            dx_trial = gg.gradient(trial, x)
            dt_trial = gg.gradient(trial, t)

        dx2_trial = g.gradient(dx_trial, x)

        del g
        del gg

        loss = (dx2_trial - dt_trial)**2

        return loss

layers = [2, 50, 50, 1]

nx = int(sys.argv[1])
nt = int(sys.argv[2])

npx = np.matrix(np.linspace(0, 1, nx)).reshape(-1, 1)
x = tf.cast(tf.convert_to_tensor(npx), tf.float32)

npt = np.matrix(np.linspace(0, 1, nt)).reshape(-1, 1)
t = tf.cast(tf.convert_to_tensor(npt), tf.float32)

Xtf, Ttf = tf.meshgrid(x, t)
X = tf.reshape(Xtf, [-1])
T = tf.reshape(Ttf, [-1])

u = heateq(layers, X, T, learning_rate=1e-3)

u.train(1000)

pred_tmp = u.predict([X, T])
pred_tmp = np.array(pred_tmp)
for i in range(len(pred_tmp)):
    if pred_tmp[i] < 0:
        pred_tmp[i] = 0
pred = np.reshape(pred_tmp, [nt, nx])

X, T = np.meshgrid(np.linspace(0, 1, nx), np.linspace(0, 1, nt))
def g(x, t):
    return np.exp(-t * np.pi**2) * np.sin(np.pi*x)
analytical = g(X, T)

abs_relativ_error = np.zeros_like(analytical)
abs_relativ_error = np.abs(analytical - pred)

print(f"Max error: {np.max(abs_relativ_error)}")

plt.figure("Prediction", figsize=(6,5))
plt.contourf(X, T, pred)
plt.xlabel("$x$")
plt.ylabel("$t$")
plt.title("Solution to the heat equation with NN")
plt.colorbar()
plt.savefig("figs/heat_nx_50_nt_50.pdf")

plt.figure("Analytical", figsize=(6,5))
plt.contourf(X, T, analytical)
plt.xlabel("$x$")
plt.ylabel("$t$")
plt.title("Analytical solution to the heat equation")
plt.colorbar()
plt.savefig("figs/analytical.pdf")

plt.figure("Absolute Error", figsize=(6,5))
plt.contourf(X, T, abs_relativ_error)
plt.xlabel("$x$")
plt.ylabel("$t$")
plt.title(f"Absolute error of the NN; max error = {np.max(abs_relativ_error)}")
plt.colorbar()
plt.savefig("figs/error_nn_nx_50_nt_50.pdf")

# plt.show()
