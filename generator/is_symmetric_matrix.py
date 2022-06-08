import numpy as np


def is_symmetric_matrix(A):
    tol = 1e-8
    return np.all(np.abs(A - A.T) < tol)
