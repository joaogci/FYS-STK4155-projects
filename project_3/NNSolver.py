import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Dense

class DiffEqNet(tf.keras.Sequential):
    """
    Use as parent class, the child class must contain the funtions:
    trial_func() : The trial solution, must obey initial and boundary conditions.
    cost_function() : Custom cost function depending on the equation.
    """
    def __init__(self, layers, learning_rate = 0.001):
        super(DiffEqNet, self).__init__()

        # Input layer
        self.add(Dense(layers[0], activation = 'linear'))

        # Hidden layers
        for layer in layers[1:-1]:
            self.add(Dense(layer, activation = 'sigmoid'))

        # Output layer
        self.add(Dense(layers[-1], activation = 'linear'))

        self.optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)
        
        self.error = list()

    @tf.function
    def gradient(self):
        with tf.GradientTape() as tape:
            tape.watch(self.trainable_variables)
            loss = self.cost_function(*self.var)

        #self.grad = tape.gradient(loss, self.trainable_variables)
        grad = tape.gradient(loss, self.trainable_variables)
        
        del tape

        return loss, grad

    @tf.function
    def update(self):
        loss, grad = self.gradient()
        self.optimizer.apply_gradients(zip(grad, self.trainable_variables))
        return loss

    #@tf.function   # Doesnt work, stalls the program for some reason
    def train(self, epochs = 5000):
        for i in range(epochs):
            print(f"{i+1: 5d}/{epochs: 5d}", end = '\r')
            #self.gradient()
            loss = self.update()
            self.error.append(tf.reduce_mean(loss).numpy())

        self.trained = True
        

    @tf.function
    def predict(self, var):
        if self.trained == False:
            print("Warning: Network not trained!")
        pred = self.trial_func(*var)

        return pred

if __name__ == "__main__":

    class expsolvenet(DiffEqNet):
        def __init__(self, layers, gamma, f0, x, learning_rate = 0.001):
            super(expsolvenet, self).__init__(layers, learning_rate)

            self.gamma = gamma
            self.f0 = f0
            self.x = x

            self.var = tuple([self.x])
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

    """
    u'(x) = -gamma*u(x)
    u(0) = 10, gamma = 2
    """
    
    x = np.matrix(np.linspace(0, 1, 1000)).reshape(-1, 1)
    x = tf.cast(tf.convert_to_tensor(x), tf.float32)

    u0 = 10
    gamma = 2

    layers = [1, 30, 30 , 1]
    u = expsolvenet(layers, gamma, u0, x)

    u.train(5000)
    pred = u.predict(tuple([x]))
    analytical = lambda x : u0*np.exp(-gamma*x)

    plt.title("Original x, same as trained interval")
    plt.plot(x, pred)
    plt.plot(x, analytical(x))
    plt.show()
