
# Runs layers_comparison.py and activation_fn_comparisons.py sequentially
from time import time
start = time()

from layers_comparison import *
print('\n\n\n')

from activation_fn_comparisons import *
print('\n\n\n')

print('Done! Took', int((time()-start)/60), 'minutes for all computations.')
plt.show()
