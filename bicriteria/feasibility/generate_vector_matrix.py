import itertools
import time

import numpy as np
import scipy.special


def generate_vector_matrix(subset_size_list, length, logfile):
    lattice_size = subset_size_list.size

    tstart = time.time()
    ## generate all combinations of vectors with repetitions
    _list = range(lattice_size)
    combinations_with_replacement_size = int(scipy.special.binom(lattice_size + length - 1, length))
    comb_time = time.time() - tstart  # record time for generating combinations

    vec_time_start = time.time()
    ## create a matrix with all vectors
    vector_matrix = np.zeros((combinations_with_replacement_size, lattice_size), dtype=np.uint8)
    for i, row in enumerate(itertools.combinations_with_replacement(_list, length)):
        np.add.at(vector_matrix[i], np.array(row), 1)
    # end for
    vec_time = time.time() - vec_time_start

    vec_proc_start = time.time()
    feasible_row_indices = np.where((vector_matrix <= subset_size_list).all(axis=1))[0]
    vector_matrix_processed = vector_matrix[feasible_row_indices, :]
    vec_proc_time = time.time() - vec_proc_start

    logfile.write("generate-vector-matrix: [vec-total: %d, vec-processed: %d]" % \
                  (vector_matrix.shape[0], vector_matrix_processed.shape[0]))
    logfile.write(" [comb-time: %.2fs, vec-time:%.2fs, vec-proc-time: %.2fs] [total: %.2fs]\n" % \
                  (comb_time, vec_time, vec_proc_time, time.time() - tstart))
    logfile.flush()

    return vector_matrix_processed.transpose()
