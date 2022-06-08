import numpy as np


def get_distance_matrix(n, seed=123456789):
    np.random.seed(seed)
    # matrix generated from uniform distribution
    # NOTE: distance function is metric
    A = np.random.uniform(low=0.5, high=1.0, size=(n, n))
    A_symm = np.tril(A) + np.tril(A, -1).T
    np.fill_diagonal(A_symm, 0.0)
    return A_symm
