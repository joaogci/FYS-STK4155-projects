from Models import Models
import numpy as np
import matplotlib.pyplot as plt

# Generate test data to run sample cases for OLS
count = 100
degree = 2
x = np.random.rand(count, 1)
y = 5*x*x + 2 + 0.1*np.random.randn(count, 1)

# Run models
linreg = Models(verbose=False)
X = linreg.design_matrix(x, degree)

#Split data
X_train, X_test, y_train, y_test = linreg.tt_split(X, y, split=0.25)

pred_train, beta = linreg.ols(X_train, y_train, pseudo_inverse=True)
pred_test = X_test @ beta
pred_full = X @ beta

# Show errors
print('MSE Train:',     linreg.mse(y_train, pred_train))
print('MSE Test:',      linreg.mse(y_test, pred_test))
print('MSE Overall:',   linreg.mse(y, pred_full))

# Show data & prediction
plt.plot(X_train[:,1], y_train ,'k+', label='Training data')
plt.plot(X_test[:,1], y_test ,'r+', label='Testing data')
plt.plot(np.sort(x, 0), np.sort(pred_full, 0), 'b-', label='Prediction')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title('Ordinary Least Squares predictions')
plt.legend()
plt.show()
