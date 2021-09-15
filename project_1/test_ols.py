from Models import Models
import numpy as np
import matplotlib.pyplot as plt

# Generate test data to run sample cases for OLS
count = 100
degree = 2
x = np.random.rand(count, 1)
y = 5*x*x + 2 + 0.1*np.random.randn(count, 1)

linreg = Models(x, y, verbose=False)
linreg.design_matrix(degree)
linreg.ols(pseudo_inverse=True)
linreg.print_error_estimates('regular OLS')

linregSVD = Models(x, y, verbose=False)
linregSVD.design_matrix(degree)
linregSVD.ols_svd()
linregSVD.print_error_estimates('SVD')

# Show data & prediction
linreg.plot(name='Standard OLS', colour='b')
linregSVD.plot(add_data=False, name='SVD OLS', colour='r-')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title('Ordinary Least Squares predictions')
plt.show()
