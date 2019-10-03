import matplotlib.pyplot as plt
import numpy as np

with open("r.txt") as file_f:
    r = np.array([row.strip() for row in file_f], dtype='float')

plt.plot(r, color = 'blue', label = 'Residual')
plt.legend()
plt.show()

