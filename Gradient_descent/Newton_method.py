# Minimize nonlinear functionality by Newton's method.
# f(x) = exp (x1 + 3x2 - 0.1) + exp (x1 - 3x2 - 0.1) + exp (−x1 - 0.1)
# The initial approximation x0 = (-1; 2).
# Draw the contour lines of the logarithm of the values of the 
# target functional log(f) and the point {x^n} on the plane.


import numpy as np
import math
import matplotlib.pyplot as plt

def f(u):
     return math.exp(u[0]+3*u[1] - 0.1) + math.exp(u[0]-3*u[1] - 0.1) + math.exp(-u[0]-0.1)

def grad(x1, x2):
    return np.array([math.exp(x1+3*x2 - 0.1) + math.exp(x1-3*x2 - 0.1) - math.exp(-x1-0.1),
            3*math.exp(x1 + 3 * x2 - 0.1) - 3*math.exp(x1 - 3 * x2 - 0.1)]).reshape(2,1)

def invH(x1, x2):
    matrix = np.matrix([[math.exp(x1+3*x2 - 0.1) + math.exp(x1-3*x2 - 0.1) + math.exp(-x1-0.1),
                         3 * math.exp(x1 + 3 * x2 - 0.1) - 3 * math.exp(x1 - 3 * x2 - 0.1)],
                        [3*math.exp(x1 + 3 * x2 - 0.1) - 3*math.exp(x1 - 3 * x2 - 0.1),
                         9*math.exp(x1+3*x2 - 0.1)+9*math.exp(x1-3*x2 - 0.1)]])
    return np.linalg.inv(matrix)

def extr():
    u = []
    u.append(np.array([-1,2]).reshape(2,1))
    k = 0
    while ((np.linalg.norm(grad(u[k][0], u[k][1]))) > 0.0001):
        u.append(u[k] - np.dot(invH(u[k][0], u[k][1]), grad(u[k][0], u[k][1])))
        k = k + 1

    return u, u[k-1]

u, ans = extr()
print(ans)
print(f(ans))
xs = list(map(lambda el: [el[0,0], el[1,0]], u))
x1res = list(map(lambda el: el[0], xs))
x2res = list(map(lambda el: el[1], xs))
plt.plot(x1res, x2res, marker='o')
plt.xlabel("x1")
plt.ylabel("x2")
delta = 0.02
x1 = np.arange(-2.0, 2.0, delta)
x2 = np.arange(-2.0, 2.0, delta)
X, Y = np.meshgrid(x1, x2)
Z = np.log(np.exp(X + 3 * Y - 0.1) + np.exp(X - 3 * Y - 0.1) + np.exp(-X - 0.1))
plt.contour(X, Y, Z, levels=15, linestyles='dotted')

plt.show()
