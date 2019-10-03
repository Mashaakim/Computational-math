import matplotlib.pyplot as plt
import numpy as np

with open("x.txt") as file_f:
    x1 = np.array([row.strip() for row in file_f], dtype='float')

with open("x2.txt") as file_f:
    x2 = np.array([row.strip() for row in file_f], dtype='float')

with open("solution.txt") as file_f:
    trx = np.array([row.strip() for row in file_f], dtype='float')

plt.plot(x1, color = 'black', label = 'First method solution')
plt.plot(x2, 'r+', color = 'green', label = 'Second method solution')
plt.plot(trx, color = 'yellow', label = 'Exact solution')
plt.legend()
plt.show()

with open("graph1.txt") as file_f:
    gr1 = np.array([row.strip() for row in file_f], dtype='float')

with open("graph2.txt") as file_f:
    gr2 = np.array([row.strip() for row in file_f], dtype='float')

plt.plot(gr1, 'r', color = 'blue', label = 'Relative residual first method')
plt.plot(gr2, color = 'green', label = 'Relative residual second method')
plt.legend()
plt.show()
