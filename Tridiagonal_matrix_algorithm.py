# Write a function that solves the three-diagonal SLAE by the Tridiagonal matrix algorithm. 
# Write a program solving SLAE Ax = f using this function


import numpy as np

with open("a_diag10.txt") as file_a:
    a = np.array([row.strip() for row in file_a], dtype='float')

with open("c_diag10.txt") as file_c:
    c = np.array([row.strip() for row in file_c], dtype='float')

with open("b_diag10.txt") as file_b:
    b = np.array([row.strip() for row in file_b], dtype='float')

with open("r10.txt") as file_r:
    r = np.array([row.strip() for row in file_r], dtype='float')

def equation(a, c, b, r):

    n = c.shape[0]
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

    # print(x)
    return alpha, beta, x


alpha, beta, x = equation(a, c, b, r)
n = c.shape[0]
x = x.reshape(n, 1)
with open('x2.txt','wb') as f:
    for line in x:
        np.savetxt(f, line)

alpha = alpha.reshape(n-1, 1)
with open('alpha.txt','wb') as f:
    for line in alpha:
        np.savetxt(f, line)

beta = beta.reshape(n, 1)
with open('beta.txt','wb') as f:
    for line in beta:
        np.savetxt(f, line)
