from Models import Models
import numpy as np
import matplotlib.pyplot as plt

# Generate test data to run sample cases for OLS
count = 10
degree = 2
x = np.random.rand(count, 1)
y = 5*x*x + 2 + 0.1*np.random.randn(count, 1)

# Run models
linreg = Models(verbose=False)
X = linreg.design_matrix(x, degree)
X_train, X_test, y_train, y_test = linreg.tt_split(X, y, split=0.25)

print("split:")
print(X_train)
print("---")
print(X_test)
print("________________")
print("")
print(y_train)
print("---")
print(y_test)
