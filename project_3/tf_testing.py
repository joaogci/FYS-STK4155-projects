import tensorflow as tf
import tensorflow.keras.backend as kb
import numpy as np

from tensorflow.keras.layers import Dense

import matplotlib.pyplot as plt

NN = tf.keras.Sequential()
NN.add(Dense(1, activation = "linear"))
NN.add(Dense(50, activation = "sigmoid"))
NN.add(Dense(1, activation = "linear"))

NN.optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)

g0 = 10
gamma = 2

x = np.matrix(np.linspace(0, 1, 10)).reshape(-1, 1)
#print(x.shape)
x = tf.cast(tf.convert_to_tensor(x), tf.float32)

def g_trial(x, NN):
    #print("æææææ")
    #return g0 + tf.math.multiply(x, NN(x, training = False))
    #print(x)
    #print("--------------")
    #print(NN(x, training = False))
    #print("--------------")
    return g0 + x * NN(x, training = False)

def g(x, NN):
    return -gamma*g_trial(x, NN)

def cost_function(x, NN):
    with tf.GradientTape() as tape:
        tape.watch(x)
        gt = tf.convert_to_tensor(g_trial(x, NN))
    d_gt = tape.gradient(gt, x)
    del tape
    #print(type(d_gt))

    #tensorg = tf.convert_to_tensor(g(x, NN))
    #print(type(g(x, NN)))
    
    return (d_gt - g(x, NN))**2

def gradient(NN):
    with tf.GradientTape() as tape:
        tape.watch(NN.trainable_variables)
        loss = cost_function(x, NN)
    d_loss = tape.gradient(loss, NN.trainable_variables)
    del tape

    return d_loss

def update(grad):
    #print(list(zip(grad, NN.trainable_variables))[0])
    NN.optimizer.apply_gradients(zip(grad, NN.trainable_variables))

def train(epochs):
    g_trial(x, NN)
    for i in range(epochs):
        grad = gradient(NN)
        update(grad)
        print(f"{i}/{epochs}", end="\r")
        #exit(1)
#print("Joao")

#print(cost_function(x, NN))
train(5000)
print(g_trial(x, NN))

g_anal = lambda x: g0*np.exp(-gamma * x)

plt.figure()
plt.plot(x, g_trial(x, NN), label="NN")
plt.plot(x, g_anal(x), label="analytic")
plt.legend()
plt.show()

