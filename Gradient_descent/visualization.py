import numpy as np
import matplotlib.pyplot as plt

delta = 0.02
x1 = np.arange(-2.0, 2.0, delta)
x2 = np.arange(-2.0, 2.0, delta)
X, Y = np.meshgrid(x1, x2)
Z1 = np.log(np.exp(X + 3 * Y - 0.1) + np.exp(X - 3 * Y - 0.1) + np.exp(-X - 0.1))

plt.contour(X, Y, Z1)
plt.show()