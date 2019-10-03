# Solve a system of nonlinear algebraic equations using the multidimensional 
# Newton method:
# x^3 − 10x + y − z + 3 = 0,
# y^3 + 10y − 2x − 2z − 5 = 0,
# x + y − 10z + 2sin(x) + 5 = 0,
# for the initial approximation (x, y, z)0 = (1,1,1).

import numpy as np
from sympy import *
import math

def diffs(expr, params, values):
    res = []
    for p in params:
        res.append(diff(expr, p).evalf(subs=values))

    return res

def F(u):
    vector = np.zeros(3)

    vector[0] = u[0]**3-10*u[0]+u[1]-u[2]+3
    vector[1] = u[1]**3+10*u[1]-2*u[0]-2*u[2]-5
    vector[2] = u[0]+u[1]-10*u[2]+2*math.sin(u[0])+5

    return vector

def Jacobi(u):
    x,y,z = symbols('x y z')
    values = {x:u[0], y:u[1], z:u[2]}
    expressions = [x**3-10*x+y-z+3, y**3+10*y-2*x-2*z-5, x+y-10*z+2*sin(x)+5]

    matrix = [diffs(expr, [x, y, z], values) for expr in expressions]
    Y = np.array(matrix, dtype='float')

    return np.linalg.inv(Y)

def NewtonMethod():
    u = []
    u.append(np.array([1,1,1]))
    u.append((u[0] - Jacobi(u[0]).dot(F(u[0]))).flatten()) #u[1]
    r = []
    r.append(np.linalg.norm(u[1]-u[0]))
    k = 1
    while (r[k-1] >= 10**(-15)):
        u.append((u[k] - Jacobi(u[k]).dot(F(u[k]))).flatten())
        r.append(np.linalg.norm(u[k+1] - u[k]))
        k+=1

    return u.pop(), r

u, r = NewtonMethod()
u = u.reshape(1,3)
# x,y,z = symbols('x y z')
# print(nsolve([x**3-10*x+y-z+3, y**3+10*y-2*x-2*z-5, x+y-10*z+2*sin(x)+5], [x,y,z], [1,1,1]))
with open('r.txt', 'w') as f:
    for line in r:
        f.write("%s\n" % line)
with open('u.txt', 'wb') as f:
    for line in u:
        np.savetxt(f, line)
