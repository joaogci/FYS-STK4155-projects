from Models import Models
import numpy as np
import matplotlib.pyplot as plt

# Generate test data to run sample cases for OLS
count = 100
degree = 2
x = np.random.rand(count, 1)
y = 5*x*x + 2 + 0.1*np.random.randn(count, 1)

# Run models
linreg = Models(x, y, verbose=False)
linreg.design_matrix(degree)

# Split data
linreg.tt_split(split=0.25)

# Prediction
linreg.ols(pseudo_inverse=True)

# Show errors
print('MSE Train:',     linreg.mse(linreg.y_train, linreg.prediction_train))
print('MSE Test:',      linreg.mse(linreg.y_test, linreg.prediction_test))
print('MSE Overall:',   linreg.mse(linreg.y, linreg.prediction))

# Show data & prediction
plt.plot(linreg.X_train[:,1], linreg.y_train ,'k+', label='Training data')
plt.plot(linreg.X_test[:,1], linreg.y_test ,'r+', label='Testing data')
plt.plot(np.sort(x, 0), np.sort(linreg.prediction, 0), 'b-', label='Prediction')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title('Ordinary Least Squares predictions')
plt.legend()
plt.show()
