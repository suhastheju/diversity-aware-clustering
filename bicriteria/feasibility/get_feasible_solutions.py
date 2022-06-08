import time

import numpy as np

from feasibility.matrix_multiplication import matrix_multiplication


def get_feasible_solutions(processed_matrix, vector_matrix, requirements, logfile):
    tstart = time.time()
    res_matrix = matrix_multiplication(processed_matrix, vector_matrix, logfile)
    matmul_time = time.time() - tstart

    feasible_row_indices = np.where((res_matrix.transpose() >= requirements).all(axis=1))[0]

    feasible_solutions = vector_matrix.transpose()[feasible_row_indices, :]
    logfile.write("get-feasible-solutions: [matrix-mul: %.2fs, total:%.2fs]" % \
                  (matmul_time, time.time() - tstart))
    logfile.write(" [total: %d, feasible: %d]\n" % \
                  (res_matrix.shape[1], feasible_solutions.shape[0]))
    logfile.flush()
    return feasible_solutions
