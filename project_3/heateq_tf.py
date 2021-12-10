import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from diffeqnet import DiffEqNet

'''
class expsolvenet(DiffEqNet):
    def __init__(self, layers, gamma, f0, x, learning_rate = 0.001):
        super(expsolvenet, self).__init__(layers, learning_rate)

        self.gamma = gamma
        self.f0 = f0
        self.x = x

        # Needs to be called in order to initialize weights and biases
        self.trial_func(x)

    @tf.function
    def func(self, x):
        return - self.gamma * self.trial_func(x)

    @tf.function
    def trial_func(self, x):
        return self.f0 + x * self(x, training = False)

    @tf.function
    def cost_function(self, x):
        with tf.GradientTape() as tape:
            tape.watch(x)
            f = tf.convert_to_tensor(self.trial_func(x))
        df = tape.gradient(f, x)

        del tape

        loss = (df - self.func(x))**2

        return loss
'''


"""
u_xx = u_t 
"""

class heateq(DiffEqNet):
    def __init__(self, layers, initial_func, x, t, learning_rate=0.001):
        super(heateq, self).__init__(layers, learning_rate)

        self.x = x
        self.t = t

        self.xt = tf.stack([x, t], axis = 1)

        self.var = tuple([x, t])

        self.initial_func = initial_func

    #@tf.function
    def trial_func(self, x, t):
        return self.initial_func(x)*(1 - t) + x*(1-x)*t*\
                tf.squeeze(self(self.xt, training = False))


    #@tf.function
    def cost_function(self, x, t):
        with tf.GradientTape(persistent = True) as g:
            g.watch(x)
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

def init_func(x):
    return tf.sin(tf.constant(np.pi)*x)

layers = [2, 30, 30, 1]

npx = np.matrix(np.linspace(0, 1, 100)).reshape(-1, 1)
x = tf.cast(tf.convert_to_tensor(npx), tf.float32)

npt = np.matrix(np.linspace(0, 1, 100)).reshape(-1, 1)
t = tf.cast(tf.convert_to_tensor(npt), tf.float32)

Xtf, Ttf = tf.meshgrid(x, t)
X = tf.reshape(Xtf, [-1])
T = tf.reshape(Ttf, [-1])

u = heateq(layers, init_func, X, T, learning_rate=1e-4)

u.train(5000)

pred = tf.reshape(u.predict([X, T]), [100, 100])

#print(pred)

x, t = np.meshgrid(npx, npt)

def g(x, t):
    return np.exp(-np.pi**2*t) * np.sin(np.pi*x)

analytical = g(x, t)
#tf_anal = tf.convert_to_tensor(analytical)
#diff = tf_anal - pred

#print(analytical)

plt.figure("Prediction")
plt.contourf(Xtf, Ttf, pred*100)
plt.colorbar()

plt.figure("anal")
plt.contourf(x, t, analytical)
plt.colorbar()

plt.show()
