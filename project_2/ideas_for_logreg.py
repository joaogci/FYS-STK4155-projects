import numpy as np

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

from NeuralNetwork.Model import Model
from NeuralNetwork.Layer import HiddenLayer, OutputLayer
from NeuralNetwork.activation_function.Sigmoid import Sigmoid
from NeuralNetwork.activation_function.LeakyReLU import LeakyReLU
from NeuralNetwork.activation_function.ReLU import ReLU
from NeuralNetwork.activation_function.ELU import ELU
from NeuralNetwork.cost_function.LogisticRegression import LogisticRegression

import pickle



def p(z):
    return 1/(1+np.exp(-z))

def binary_output(output):
    return output.round(0)

seed = 1
rng = np.random.default_rng(np.random.MT19937(seed=seed))

cancerdata = load_breast_cancer() #(569, 30)

data = cancerdata['data']
target = cancerdata['target']

X_train, X_test, y_train, y_test = \
        train_test_split(data, target, test_size=0.2, random_state=seed)

""" Scaling our data """
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.fit_transform(X_test)

""" Neural net """
nn_input_train = X_train_s
nn_input_test = X_test_s
nn_output_train, nn_output_test = np.matrix(y_train).T, np.matrix(y_test).T

nn = Model(X_train.shape[1], cost_function=LogisticRegression(nn_input_train, nn_output_train, nn_input_test, nn_output_test))
nn.add_layer(HiddenLayer(20, activation_function=ELU()))
nn.add_layer(HiddenLayer(20, activation_function=ELU()))
nn.add_layer(OutputLayer(1, activation_function=Sigmoid()))

# Train network with grid search
nn.grid_train_sgd(nn_input_train, nn_output_train, nn_input_test, nn_output_test, filename='logreg_results', plot=True, initial_learning_rate=np.arange(0.05, 0.2, 0.05), final_learning_rate=None, regularization=np.arange(0, 0.01, 0.0025), epochs=100)

# Obtaining results out of the file:
# with open('results/logreg_results.pickle', 'rb') as handle:
#     result = pickle.load(handle)
#     print(result)




LogReg = LogisticRegression(max_iter = 10000, penalty='none',random_state=seed)
LogReg.fit(X_train, y_train)

accuracy = LogReg.score(X_test, y_test)

print(f"Accuracy of sklearn LogisticRegresison: {accuracy:.5g}")

y_train = np.array([y_train]).reshape(-1, 1)
y_test = np.array([y_test]).reshape(-1, 1)

""" Gradient decent/Newton's method """

n_features = len(X_train[0])
# Initial guess for gradient decent/Newton's method
beta = 0.01 * rng.normal(size=(n_features, 1))

learning_rate = 0.0002
max_iterations = 10000
for i in range(max_iterations):
    z = X_train_s.dot(beta)

    W = np.diag( (p(z)*(1-p(z))).reshape(-1) )

    grad_C = X_train_s.T @ (y_train - p(z)) # * 1/len(X_train)

    Hess_C = X_train_s.T @ W @ X_train_s


    # beta = beta + np.linalg.pinv(Hess_C) @ grad_C
    beta = beta + learning_rate * grad_C      # Const learn_rate, no overflow


# print("Newton iteration")
print("Gradient Descent")
pred = binary_output(p(X_test_s.dot(beta)))
print(np.sum(pred == y_test),"/", len(y_test))

""" Stochastic Gradient Descent (sklearn) """
learning_rate = 0.00025

#sklearn
SGDLogReg = SGDClassifier(loss = 'log', max_iter = 10000, alpha=0, learning_rate='constant', eta0 = learning_rate)
SGDLogReg.fit(X_train, y_train)
SGDaccuracy = SGDLogReg.score(X_test, y_test)

print(f"Accuracy SGD Classifier: {SGDaccuracy}")


""" Stochastic Gradient Descent (our own) """
epochs = 10000
batch_size = 1
num_batches = int(n_features/batch_size) 
theta = rng.normal(size=(n_features, 1))  # Same as beta, but we define a new one since we are doing a new method

tol = 1e-6
for epoch in range(epochs):
    for batch in range(num_batches):
        indeces = np.random.randint(0, high = n_features, size = batch_size)
      
        X_b = X_train_s[indeces]
        y_b = y_train[indeces]

        z = X_b.dot(theta)

        grad =  X_b.T @ (y_b - p(z)) # * (1/batch_size)
        
        if np.linalg.norm(grad) < tol:
            break

        theta = theta + learning_rate * grad


test_prob = p(X_test_s.dot(theta))

pred_SGD = binary_output(test_prob)
sk_pred_SGD = SGDLogReg.predict(X_test)

# print("Compare our SGD to SKLearn:")
# print(f"{np.sum(pred_SGD==sk_pred_SGD)}/{len(pred_SGD)}")
pred_SGD_acc = np.sum(pred_SGD == y_test)
print("Accuracy, own SGD logisitc regression:", pred_SGD_acc, "/", len(y_test))

