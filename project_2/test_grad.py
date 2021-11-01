
import autograd.numpy as np
from autograd import elementwise_grad
import matplotlib.pyplot as plt

def f(a, x):
    return np.sin(np.pi * x) + a

n = 1000
x = np.linspace(-1, 1, n)

grad_f = elementwise_grad(f, 1)

plt.plot(x, f(1, x), '-r')
plt.plot(x, grad_f(1, x)/np.pi, '--b')

plt.show()
