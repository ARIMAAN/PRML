
import numpy as np

def eigen(A):
    eigenvalues, eigenvectors = np.linalg.eig(A)

    # Eigenvectors are already arranged as columns
    V = eigenvectors

    # Put eigenvalues along the diagonal
    D = np.diag(eigenvalues)

    return V, D
def matrix_inverse(A):
    A_inverse = np.linalg.inv(A)
    return A_inverse
