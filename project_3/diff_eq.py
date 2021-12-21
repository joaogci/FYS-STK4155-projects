import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from NNSolver import DiffEqNet

class ode1(DiffEqNet):
    def __init__(self, layers, x, learning_rate = 1e-3):
        super(ode1, self).__init__(layers, learning_rate)

        self.var = tuple([x])

    @tf.function
    def trial_func(self, x):
        """ Domain: [0.1, 1], Condition: f(0.1) = 20.1 """
        return 20.1 + (x - 0.1)*self(x, training = False)

    @tf.function
    def cost_function(self, x):
        with tf.GradientTape() as g:
            g.watch(x)
            trial = self.trial_func(x)

        dx_trial = g.gradient(trial, x)

        del g

        loss = (dx_trial - (2*x - trial)/x)**2

        return loss
    
    def analytical_solution(self, x):
        return x + 2/x

class ode2(DiffEqNet):
    def __init__(self, layers, x, learning_rate = 1e-3):
        super(ode2, self).__init__(layers, learning_rate)

        self.var = tuple([x])

    @tf.function
    def trial_func(self, x):
        """ Domain: [0.1, 1], Condition: f(0.1) = 2.1/sin(0.1) """
        return 2.1/tf.sin(0.1) + (x - 0.1)*self(x, training = False)

    @tf.function
    def cost_function(self, x):
        with tf.GradientTape() as g:
            g.watch(x)
            trial = self.trial_func(x)

        dx_trial = g.gradient(trial, x)

        del g

        loss = (dx_trial - (1 - trial*tf.cos(x))/tf.sin(x))**2

        return loss
    
    def analytical_solution(self, x):
        return (x + 2)/np.sin(x)

class ode3(DiffEqNet):
    def __init__(self, layers, x, learning_rate = 1e-3):
        super(ode3, self).__init__(layers, learning_rate)

        self.var = tuple([x])

    @tf.function
    def trial_func(self, x):
        """ Domain: [0, 1], Condition: f(0) = 0 """
        return x*self(x, training = False)

    @tf.function
    def cost_function(self, x):
        with tf.GradientTape() as g:
            g.watch(x)
            trial = self.trial_func(x)

        dx_trial = g.gradient(trial, x)

        del g

        loss = (dx_trial - (-0.2*trial + tf.exp(-x/5)*tf.cos(x)))**2

        return loss
    
    def analytical_solution(self, x):
        return np.exp(-x/5)*np.sin(x)

class ode7(DiffEqNet):
    def __init__(self, layers, x, learning_rate = 1e-3):
        super(ode7, self).__init__(layers, learning_rate)

        self.var = tuple([x])

    @tf.function
    def trial_func(self, x):
        """ Domain: [0, 1], Condition: f(0) = 0, f(1) = sin(10) """
        a = tf.constant(10, tf.float32)
        return x*tf.sin(a) + x*(x - 1)*self(x, training = False)

    @tf.function
    def cost_function(self, x):
        with tf.GradientTape(persistent = True) as g:
            g.watch(x)

            with tf.GradientTape(persistent = True) as gg:
                gg.watch(x)
                trial = self.trial_func(x)
                
            dx_trial = gg.gradient(trial, x)

        dx2_trial = g.gradient(dx_trial, x)

        del g
        del gg

        loss = (dx2_trial - (-100*trial))**2

        return loss
    
    def analytical_solution(self, x):
        return np.sin(10*x)

class ode8(DiffEqNet):
    def __init__(self, layers, x, learning_rate = 1e-3):
        super(ode8, self).__init__(layers, learning_rate)

        self.var = tuple([x])

    @tf.function
    def trial_func(self, x):
        """ Domain: [0, 1], Condition: f(0) = 1, f(1) = 0 """
        return (1 - x) + x*(x - 1)*self(x, training = False)

    @tf.function
    def cost_function(self, x):
        with tf.GradientTape(persistent = True) as g:
            g.watch(x)

            with tf.GradientTape(persistent = True) as gg:
                gg.watch(x)
                trial = self.trial_func(x)

            dx_trial = gg.gradient(trial, x)

        dx2_trial = g.gradient(dx_trial, x)

        del g
        del gg

        loss = (x * dx2_trial - ((x-1)*dx_trial - trial))**2

        return loss
    
    def analytical_solution(self, x):
        return 1 - x

class ode9(DiffEqNet):
    def __init__(self, layers, x, learning_rate = 1e-3):
        super(ode9, self).__init__(layers, learning_rate)

        self.var = tuple([x])

    @tf.function
    def trial_func(self, x):
        """ Domain: [0, 1], Condition: f(0) = 0, f(1) = sin(1)/e^0.2 """
        a = tf.constant(1, tf.float32)
        b = tf.constant(0.2, tf.float32)
        return x*tf.sin(a)/tf.exp(b) + x*(x - 1)*self(x, training = False)

    @tf.function
    def cost_function(self, x):
        with tf.GradientTape(persistent = True) as g:
            g.watch(x)

            with tf.GradientTape(persistent = True) as gg:
                gg.watch(x)
                trial = self.trial_func(x)

            dx_trial = gg.gradient(trial, x)

        dx2_trial = g.gradient(dx_trial, x)

        del g
        del gg

        loss = (dx2_trial - (-0.2*dx_trial - trial - 0.2*tf.exp(-x/5)*tf.cos(x)))**2

        return loss
    
    def analytical_solution(self, x):
        return np.exp(-x/5)*np.sin(x)

class nlode1(DiffEqNet):
    def __init__(self, layers, x, learning_rate = 1e-3):
        super(nlode1, self).__init__(layers, learning_rate)

        self.var = tuple([x])

    @tf.function
    def trial_func(self, x):
        """ Domain: [1, 4], Condition: f(1) = 1 """
        return x + (x - 1)*self(x, training = False)

    @tf.function
    def cost_function(self, x):
        with tf.GradientTape(persistent = True) as g:
            g.watch(x)
            trial = self.trial_func(x)

        dx_trial = g.gradient(trial, x)

        del g

        loss = (dx_trial - (1/(2*trial)))**2

        return loss
    
    def analytical_solution(self, x):
        return np.sqrt(x)

class nlode2(DiffEqNet):
    def __init__(self, layers, x, learning_rate = 1e-3):
        super(nlode2, self).__init__(layers, learning_rate)

        self.var = tuple([x])

    @tf.function
    def trial_func(self, x):
        """ Domain: [1, 2], Condition: f(1) = 1 + sin(1) """
        a = tf.constant(1, tf.float32)
        return x*(1 + tf.sin(a)) + x*(x - 1)*self(x, training = False)

    @tf.function
    def cost_function(self, x):
        with tf.GradientTape(persistent = True) as g:
            g.watch(x)

            with tf.GradientTape(persistent = True) as gg:
                gg.watch(x)
                trial = self.trial_func(x)

            dx_trial = gg.gradient(trial, x)

        dx2_trial = g.gradient(dx_trial, x)

        del g
        del gg

        loss = (dx2_trial**2 - (-tf.math.log(trial) + tf.cos(x)**2 \
                + 2*tf.cos(x) + 1 + tf.math.log(x + tf.sin(x))))**2

        return loss
    
    def analytical_solution(self, x):
        return x + np.sin(x)

class nlode3(DiffEqNet):
    def __init__(self, layers, x, learning_rate = 1e-3):
        super(nlode3, self).__init__(layers, learning_rate)

        self.var = tuple([x])

    @tf.function
    def trial_func(self, x):
        """ Domain: [1, 2], Condition: f(1) = 0 """
        return x - 1 + (x - 1)*self(x, training = False)

    @tf.function
    def cost_function(self, x):
        with tf.GradientTape(persistent = True) as g:
            g.watch(x)

            with tf.GradientTape(persistent = True) as gg:
                gg.watch(x)
                trial = self.trial_func(x)

            dx_trial = gg.gradient(trial, x)

        dx2_trial = g.gradient(dx_trial, x)

        del g
        del gg

        loss = (dx2_trial*dx_trial + 4/(x**3))**2

        return loss
    
    def analytical_solution(self, x):
        return np.log(x**2)

if __name__ == "__main__":

    tf.random.set_seed(1826)

    layers = [1, 50, 50, 1]

    npx011 = np.matrix(np.linspace(0.1, 1, 1000)).reshape(-1, 1)
    x011 = tf.cast(tf.convert_to_tensor(npx011), tf.float32)

    npx01 = np.matrix(np.linspace(0, 1, 1000)).reshape(-1, 1)
    x01 = tf.cast(tf.convert_to_tensor(npx01), tf.float32)

    npx12 = np.matrix(np.linspace(1, 2, 1000)).reshape(-1, 1)
    x12 = tf.cast(tf.convert_to_tensor(npx12), tf.float32)
    
    npx14 = np.matrix(np.linspace(1, 4, 1000)).reshape(-1, 1)
    x14 = tf.cast(tf.convert_to_tensor(npx14), tf.float32)

    def printscore(eq, x, epochs, lr = 1e-3, plot = False):
        f = eq(layers, x, lr)
        f.train(epochs)

        mse, r2, max_err = f.score()

        print(f"Name: {eq.__name__}")
        print(f"Epochs: {epochs}")
        print(f"MSE: {mse}")
        print(f"R2: {r2}")
        print(f"Max Error: {max_err}")

        if plot:
            pred = np.array(f.predict(x))
            np_x = np.array(x)
            analytical = f.analytical_solution(np_x)

            plt.figure(f"{eq.__name__}, Epochs: {epochs}")

            plt.title(f"Epochs: {epochs}")
            plt.plot(np_x, pred, label = "prediction")
            plt.plot(np_x, analytical, label = "analytical", ls = "--")
            plt.legend()

    printscore(ode1, x011, 1000, plot = True)
    printscore(ode1, x011, 5000, plot = True)
    printscore(ode1, x011, 10000, plot = True)

    printscore(ode2, x011, 1000, plot = True)
    printscore(ode2, x011, 5000, plot = True)
    printscore(ode2, x011, 10000, plot = True)

    printscore(ode3, x01, 1000, plot = True)

    printscore(ode7, x01, 1000, plot = True)
    printscore(ode7, x01, 5000, plot = True)
    printscore(ode7, x01, 10000, plot = True)
    
    printscore(ode8, x01, 1000, plot = True)

    printscore(ode9, x01, 1000, plot = True)

    printscore(nlode1, x14, 1000, plot = True)

    printscore(nlode2, x12, 1000, plot = True)
    printscore(nlode2, x12, 5000, plot = True)

    printscore(nlode3, x12, 1000, plot = True)
    printscore(nlode3, x12, 5000, plot = True)
    printscore(nlode3, x12, 10000, plot = True)

    plt.show()
