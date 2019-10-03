# Program a Fixed-point iteration WITHOUT choosing an iteration 
# parameter with a diagonal preconditioner
import numpy as np

def dot(diag, diagLow, vector):
    u = np.zeros([n,1], dtype='float')
    u[0, 0] = diag[0]*diagLow[0]*vector[1]
    u[n-1, 0] = diag[n-1]*diagLow[0]*vector[n-2]
    for i in range(1,n-2):
        u[i, 0] = diag[i]*diagLow[i]*(vector[i-1] + vector[i+1])

    return -u

def dotDiag(diag, vector):
    u = np.zeros([n, 1], dtype='float')
    for i in range(n):
        u[i][0] = vector[i]*diag[i]

    return u

def dotA(diag, diagLow, vector):
    u = np.zeros([n,1], dtype='float')
    u[0, 0] = diag[0]*vector[0] + diagLow[0]*vector[1]
    u[n-1, 0] = diag[0]*vector[n-1] + diagLow[0]*vector[n-2]
    for i in range(1,n-2):
        u[i, 0] = diagLow[i]*(vector[i-1] + vector[i+1]) + diag[i]*vector[i]

    return u

def solve(diag, diagD, diagLow, f):
    x = []
    r = []
    graph = []
    x.append(np.zeros(n))
    # Residual r[i] = f - A*x[i]
    r.append(np.squeeze(f.reshape(n, 1) - dotA(diag, diagLow, x[0])))
    norma = np.linalg.norm(r[0])
    normaTwo = norma
    graph.append(1)
    k = 1
    while (normaTwo / norma >= 0.001):
        x.append(np.squeeze(dot(diagD, diagLow, x[k-1]) + dotDiag(diagD, f)))
        r.append(np.squeeze(f.reshape(n, 1) - dotA(diag, diagLow, x[k])))
        normaTwo = np.linalg.norm(r[k])

        graph.append(normaTwo / norma)
        k += 1

    # print(k) = 1203
    return x.pop(), graph, k

with open("r10.txt") as file_f:
    f = np.array([row.strip() for row in file_f], dtype='float')

n = 1000
beta = 10
alpha = 0.01
diag = np.ndarray(n)
diagD = np.ndarray(n)
diagLow = np.ndarray(n-1)
diag[0] = beta + 2
for i in range(1, n):
    diag[i] = alpha + 2
for i in range(0, n-1):
    diagLow[i] = -1

matrixD = np.zeros([n, n], dtype='float')
for i in range(n):
    matrixD[i][i] = diag[i]

matrixD = np.linalg.inv(matrixD)
for i in range(n):
    diagD[i] = matrixD[i][i]

x, graph, k = solve(diag, diagD, diagLow, f)
x = x.reshape(n, 1)
graph = np.asarray(graph).reshape(k, 1)
with open('x2.txt', 'wb') as f:
    for line in x:
        np.savetxt(f, line)

with open('graph2.txt', 'wb') as f:
    for line in graph:
        np.savetxt(f, line)


