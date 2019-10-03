# Write a function that performs the LU decomposition of the real 
# square matrix without selecting the main element. 
# Write a program solving SLAE Ax = f using LU decomposition


import numpy as np

with open("A10.txt") as file_matrix:
    array = np.array([row.strip() for row in file_matrix], dtype='float')

matrix = array.reshape(3,3)
if (np.linalg.det(matrix) == 0):
    print('Matrix rows are linearly dependent')

def LU(matrix):

    LU_matrix = np.matrix(np.zeros([3, 3]))
    n = 3

    for k in range(n):
        for j in range(k, n):
            LU_matrix[k, j] = matrix[k, j] - (LU_matrix[k, :k] * LU_matrix[:k, j])
        for i in range(k + 1, n):
            LU_matrix[i, k] = (matrix[i, k] - LU_matrix[i, :k] * LU_matrix[:k, k]) / LU_matrix[k, k]

    L = LU_matrix.copy()
    for i in range(3):
        L[i, i] = 1
        L[i, i + 1:] = 0

    U = LU_matrix.copy()
    for i in range(1, 3):
        U[i, :i] = 0

    return L, U


L, U = LU(matrix)
# print(matrix)

with open("f10.txt") as file_vector:
    array = np.array([row.strip() for row in file_vector], dtype='float')

f = array.reshape(3,1)

def equation(L, U, f):
    y = np.matrix(np.zeros([3, 1]))
    for i in range(y.shape[0]):
        y[i, 0] = f[i, 0] - (L[i, :i] * y[:i])

    x = np.matrix(np.zeros([3, 1]))
    for i in range(1, x.shape[0] + 1):
        x[-i, 0] = (y[-i] - U[-i, -i:] * x[-i:, 0])/ U[-i, -i]

    return x

# print(equation(L, U, f))
# x1 = np.linalg.solve(matrix, f)
# print(x1)
x = equation(L, U, f)

with open('x.txt','wb') as f:
    for line in x:
        np.savetxt(f, line)


U = U.transpose()
U = U.reshape(9,1)
with open('U.txt','wb') as f:
    for line in U:
        np.savetxt(f, line)

L = L.transpose()
L = L.reshape(9,1)
with open('L.txt','wb') as f:
    for line in L:
        np.savetxt(f, line)
