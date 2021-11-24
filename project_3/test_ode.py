import autograd.numpy as np
from autograd import grad, elementwise_grad
import matplotlib.pyplot as plt

seed = 1337

def sigmoid(z):
    return 1 / (1 + np.exp(- z))

def g_trial(x, P, g0=10):
    return g0 + x * feed_forward(x, P)

def g(x, P, gamma=2, g0=10):
    return - gamma * g_trial(x, P, g0=g0)

def g_anal(x, gamma=2, g0=10):
    return g0 * np.exp(- gamma * x)

def cost_function(x, P):
    return np.mean((elementwise_grad(g_trial, 0)(x, P) - g(x, P))**2)
    
def feed_forward(x, P):
    w_hidden = P[0]
    w_output = P[1]
    
    n = np.size(x)
    # x = np.array(x)
    # print()
    # print(type(x))
    # print(x.shape)
    # print()
    x = x.reshape(-1, n)
    
    # print()
    # print(x.shape)
    # print()
    
    x_input = np.concatenate((np.ones((1, n)), x), axis=0)
    # print()
    # print(w_hidden.shape)
    # print(x_input.shape)
    # print()
    x_hidden = sigmoid(np.matmul(w_hidden, x_input))
    x_hidden = np.concatenate((np.ones((1, n)), x_hidden), axis=0)
    
    x_output = np.matmul(w_output, x_hidden)
    
    return x_output

def sovle_ode(x, n_hidden, eta, iter_max, seed=1337):
    rng = np.random.default_rng(np.random.MT19937(seed=seed))
    
    p_hidden = rng.uniform(size=(n_hidden, 2))
    p_output = rng.uniform(size=(1, n_hidden + 1))
    P = [p_hidden, p_output]
    
    grad_C_P = grad(cost_function, 1)
    
    for iter in range(1, iter_max + 1):
        grad_C = grad_C_P(x, P)
        print()
        print(grad_C)
        print()
        
        P[0] = P[0] - eta * grad_C[0]
        P[1] = P[1] - eta * grad_C[1]
        
        print(f" [ iter: {iter}/{iter_max} ] ", end='\r')
        
    return P

def forward_euler(x, f0):
    n = np.size(x)
    h = (x[-1] - x[0]) / n
    
    solution = np.zeros(n)
    solution[0] = f0
    
    for i in range(0, n - 1):
        solution[i + 1] = solution[i] - h * gamma * solution[i]
        
    return solution
    

if __name__ == '__main__':
    
    N = 5
    x = np.linspace(0, 1, N)
    g0 = 10
    gamma = 2
    
    n_hidden = 16
    iter_max = int(1e4)
    eta = 0.01
    
    P = sovle_ode(x, n_hidden=n_hidden, iter_max=iter_max, eta=eta)
    
    solution_nn = g_trial(x, P)
    solution_nn = solution_nn[0, :]
    solution_anal = g_anal(x)
    solution_euler = forward_euler(x, g0)
    
    plt.figure(1)
    
    plt.title("Neural Network vs Analytic Solution")
    plt.plot(x, solution_anal, label="Analytic")
    plt.plot(x, solution_nn, label="Neural Network")
    plt.plot(x, solution_euler, label="Forward Euler")
    plt.xlabel(r"$x$")
    plt.ylabel(r"%$g(x)$")
    plt.legend()
    
    plt.figure(2)
    
    plt.title("Relative absolute error in the solution")
    plt.plot(x, np.abs(solution_nn - solution_anal)/solution_anal, label="Neural Network")
    plt.plot(x, np.abs(solution_euler - solution_anal)/solution_anal, label="Forward Euler")
    plt.xlabel(r"$x$")
    plt.ylabel(r"$|\epsilon|$") 
    plt.legend()
    
    plt.show()




