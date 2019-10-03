# Minimize nonlinear functionality by Gradient descent.
# f(x) = exp (x1 + 3x2 - 0.1) + exp (x1 - 3x2 - 0.1) + exp (−x1 - 0.1)
# The initial approximation x0 = (-1; 2).
# Draw the contour lines of the logarithm of the values of the 
# target functional log(f) and the point {x^n} on the plane.


import numpy as np
import math
import matplotlib.pyplot as plt


def f(x1, x2):
    return math.exp(x1 + 3 * x2 - 0.1) + math.exp(x1 - 3 * x2 - 0.1) + math.exp(-x1 - 0.1)

def grad(x1, x2):
    return np.array([math.exp(x1+3*x2 - 0.1) + math.exp(x1-3*x2 - 0.1) - math.exp(-x1-0.1),
            3*math.exp(x1 + 3 * x2 - 0.1) - 3*math.exp(x1 - 3 * x2 - 0.1)]).reshape(2,1)

def min(x0, y0):
    a = -1
    b = 1
    TAOm = (a+b)/2
    L = b - a
    Fm = f(x0 - TAOm*(grad(x0,y0)[0]), y0 - TAOm*(grad(x0,y0)[1]))
    k = 0
    while (L > 0.001):
        tao1 = a + L/4
        tao2 = b - L/4
        k = k+1
        if (f(x0-tao1*(grad(x0,y0)[0]), y0-tao1*(grad(x0,y0)[1])) < Fm):
            b = TAOm
            TAOm = tao1
            Fm = f(x0 - TAOm*(grad(x0,y0)[0]), y0 - TAOm*(grad(x0,y0)[1]))
        else:
            if (f(x0-tao2*(grad(x0,y0)[0]), y0-tao2*(grad(x0,y0)[1])) < Fm):
                a = TAOm
                TAOm = tao2
                Fm = f(x0 - TAOm*(grad(x0,y0)[0]), y0 - TAOm*(grad(x0,y0)[1]))
            else:
                a = tao1
                b = tao2

        L = b - a
    print(k)

    return TAOm


def extrGrad():
    u = []
    u.append(np.array([-1, 2]).reshape(2, 1))
    k = 0
    while (np.linalg.norm(grad(u[k][0],u[k][1])) > 0.001):
        tao = min(u[k][0], u[k][1])
        print(tao)
        u.append(u[k] - tao*grad(u[k][0], u[k][1]))
        k = k + 1

    return u, u[k-1]

u, ans = extrGrad()
print(ans)
print(f(ans[0], ans[1]))

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
