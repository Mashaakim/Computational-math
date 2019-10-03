# Program fixed-point iteration with the choice of an iterative parameter and solve SLAE 𝐴𝑥 = 𝑓
# with a three-diagonal matrix 1000x1000.
# for 𝛽 = 10, 𝛼 = 0.01, and zero initial approximation. Stop the process when the relative residual ‖𝑟𝑘‖ / ‖𝑟0‖ = 1E-3 is reached.


import numpy as np

def eigenvalues(n, diag, diagLow):
    matrix = np.zeros([n, n], dtype='float')
    for i in range(n):
        matrix[i][i] = diag[i]
    for i in range(n-1):
        matrix[i+1][i] = diagLow[i]
        matrix[i][i+1] = matrix[i+1][i]

    values = np.linalg.eig(matrix)
    minValue = min(values[0])
    maxValue = max(values[0])

    return minValue, maxValue, matrix

def dot(diag, diagLow, vector):
    u = np.zeros([n,1], dtype='float')
    u[0, 0] = diag[0]*vector[0] + diagLow[0]*vector[1]
    u[n-1, 0] = diag[0]*vector[n-1] + diagLow[0]*vector[n-2]
    for i in range(1,n-2):
        u[i, 0] = diagLow[i]*(vector[i-1] + vector[i+1]) + diag[i]*vector[i]

    return u


def solve(f, diag, diagLow, n, tOpt):
    x = []
    r = []
    graph = []

    x.append(np.zeros(n))
    r.append(np.squeeze(f - dot(diag, diagLow, x[0])))
    x.append(x[0] + tOpt * r[0])

    norma = np.linalg.norm(r[0])
    normaTwo = norma
    graph.append(1)
    k = 1
    while (normaTwo / norma >= 0.001):
        r.append(np.squeeze(f - dot(diag, diagLow, x[k])))
        x.append(x[k] + tOpt * r[k])
        normaTwo = np.linalg.norm(r[k])
        k += 1
        graph.append(normaTwo / norma)

    return x.pop(), graph, k


with open("r10.txt") as file_f:
    f = np.array([row.strip() for row in file_f], dtype='float')

n = 1000
beta = 10
alpha = 0.01
f = f.reshape(n, 1)
diag = np.ndarray(n)
diagLow = np.ndarray(n-1)
diag[0] = beta + 2
for i in range(1, n):
    diag[i] = alpha + 2
for i in range(0, n-1):
    diagLow[i] = -1

lambdaMin, lambdaMax, A = eigenvalues(n, diag, diagLow)

tOpt = 2/(lambdaMin + lambdaMax)

x, graph, k = solve(f, diag, diagLow, n, tOpt)
# k = 3628
x = x.reshape(n, 1)
graph = np.asarray(graph).reshape(k, 1)
# check the solution
trueX = np.linalg.solve(A, f)
if (np.linalg.norm(x-trueX) < 1):
    with open('x1.txt', 'wb') as f:
        for line in x:
            np.savetxt(f, line)

    trueX = trueX.reshape(n, 1)
    with open('trueX.txt', 'wb') as f:
        for line in trueX:
            np.savetxt(f, line)

    with open('graph1.txt', 'wb') as f:
        for line in graph:
            np.savetxt(f, line)
