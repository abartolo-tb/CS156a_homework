"""
CS 156a: Final Exam
Anthony Bartolotta
Problem 12
"""
import numpy as np
import quadprog as qp


def kernel(x1, x2):
    # Polynomial kernel of degree 2
    k = (1 + np.dot(x1, x2))**2
    return k

def SVM(X_vec, y_vec):
    n = len(y_vec)
    # Define quadratic cost matrix
    M = np.array([ [ y_vec[i]*y_vec[j]*kernel(X_vec[i,:], X_vec[j,:]) \
                    for j in range(n)] for i in range(n)], dtype='float64')
    # Add small positive cost in each direction to avoid pathologies
    for j in range(n):
        M[j,j] += 10**(-7)
    # Define linear cost vector
    q = np.ones((n,))
    # Inequality constraints
    G = np.identity(n)
    h = np.zeros((n,))
    # Equality constraints
    A = np.reshape(y_vec, (1,n))
    b = 0
    # Package constraints
    qp_C = np.vstack([A, G]).T
    qp_b = np.hstack([b, h])
    neq = 1
    # Solve quadratic program
    a_Vec = qp.solve_qp(M, q, qp_C, qp_b, neq)[0]
    return a_Vec

def main():
    # Define data
    X_vec = np.array([ [1,0], [0,1], [0,-1], [-1,0], [0,2], [0,-2], [-2,0] ])
    y_vec = np.array([-1, -1, -1, 1, 1, 1, 1])
    # Use hard-margin SVM with polynomial kernel to find support vectors.
    a_vec = SVM(X_vec, y_vec)
    # Introduce rounding error for support vectors
    roundingError = 10**(-5)
    # Find number of support vectors.
    nSupport = sum(a_vec >= roundingError)
    print("Number of support vectors = " + repr(nSupport))
    return

main()
