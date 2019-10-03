# Implement an explicit scheme and the Crank-Nicholson scheme 
# to find an approximate solution of the heat equation

import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd


def a(n,m,tao,h):
    return (1+(m*h)**4)


def f(n,m,tao,h):
    return math.exp(-3.0*n*tao)*math.sin(3.1415*m*h)*(-3+9.869*a(n,m,tao,h))


def ExplicitMatrix():
    h = 0.05
    tao = 0.000625 #0.25*h*h 0.000625
    U = np.zeros((1600, 20))
    x = 0
    j = 0
    for i in range(20):
        U[0][i] = math.sin(3.1415*j)
        j+=h
        
    return U, tao, h


def ExplicitScheme(matrix, tao, h):
    n = 0
    m = 1
    for j in range(1599):
        for i in range(17):
            matrix[n+1][m] = matrix[n][m]+((tao/h/h))*a(n,m,tao,h)*(matrix[n][m+1]-2.0*matrix[n][m]+matrix[n][m-1])+tao*f(n,m,tao,h)
            m+=1
        m = 1
        n+=1
    return matrix


def running(a, c, b, r):
    n = len(c)
    alpha = np.ndarray(n-1)
    beta = np.ndarray(n)

    alpha[0] = b[0]/c[0]
    beta[0] = r[0]/c[0]

    for i in range(1, n-2):
        alpha[i] = b[i]/(c[i]-a[i-1]*alpha[i-1])
        beta[i] = (r[i]-a[i-1]*beta[i-1])/(c[i]-a[i-1]*alpha[i-1])

    beta[n-1] = (r[n-1]-a[n-2]*beta[n-2])/(c[n-1]-a[n-2]*alpha[n-2])

    x = np.ndarray(n)
    x[n-1] = beta[n-1]
    for i in reversed(range(n-1)):
        x[i] = beta[i] - alpha[i]*x[i+1]

    return x


def KNMatrix():
    h1 = 0.05
    tao1 = 0.02
    U1 = np.zeros((50, 20))
    j = 0
    for i in range(20):
        U1[0][i] = math.sin(3.1415*j)
        j+=0.05
        
    return U1, tao1, h1


def c_diag(tao, h):
    c = []
    n = 1
    for m in range(1, 19):
        c.append(1+(tao/h/h)*a(n,m,tao,h))
    return c

def a_diag(tao, h):
    a_list = []
    n = 1
    for m in range(2, 19):
        a_list.append(-(tao/h/h/2)*a(n,m,tao,h))
    return a_list

def b_diag(tao, h):
    b = []
    n = 1
    for m in range(1, 18):
        b.append(-(tao/h/h/2)*a(n,m,tao,h))
    return b

def r_vec(matrix, tao, h, n):
    d = []
    for m in range(1, 19):
        d.append(matrix[n][m]*(1-(tao/h/h)*a(n,m,tao,h))+(tao/h/h/2)*a(n,m,tao,h)*(matrix[n][m-1]+matrix[n][m+1])+tao*(f(n,m,tao,h)+f(n+1,m,tao,h))/2)
    return d


def CN(matrix, tao1, h1):
    a_list = a_diag(tao1,h1)
    c = c_diag(tao1,h1)
    b = b_diag(tao1,h1)
    for n in range(49):
        r = r_vec(matrix, tao1,h1,n)
        u = running(a_list,c,b,r)
        i = 0
        for m in range(1,18):
            matrix[n+1,m] = u[i]
            i+=1
    return matrix


U, tao, h = ExplicitMatrix()
U1, tao1, h1 = KNMatrix()
ExpSolve = ExplicitScheme(U, tao, h)
CNSolve = CN(U1, tao1, h1)


acc = np.ndarray(10)
t=0
for i in range(10):
    acc[i] = math.exp(-3*t)*math.sin(3.1415*0.5)
    t+=0.1


explicit = np.ndarray(10)
k=0
for i in range(10):
    explicit[i] = ExpSolve[k][10]
    k+=160


implicit = np.ndarray(10)
k=0
for i in range(10):
    implicit[i] = CNSolve[k][10]
    k+=5


index = ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0']
plt.plot(index, explicit, "bo", color = 'red', label = 'Explicit Scheme')
plt.plot(index, acc, "bo", color='green', label = 'Accurate')
plt.plot(index, implicit, "bo", label = 'CN')
plt.legend()
plt.show()


data = {'ES': explicit, 'CN': implicit, 'Exact Solution': acc}
df = pd.DataFrame(data=data, index = index)
df

