import itertools
import time

import numpy as np


def get_any_feasible_solution(processed_matrix, requirements, subset_size_list,
                              length, batch_size, logfile):
    lattice_size = subset_size_list.size
    tstart = time.time()
    _list = range(lattice_size)
    result = None
    iterator = itertools.combinations_with_replacement(_list, length)
    elements = []
    while (True):
        el = next(iterator, None)
        if el is not None:
            elements.append(list(el))
        # end if

        if (el is None or len(elements) == batch_size) and len(elements) != 0:
            comb_matrix = np.array(elements, dtype='uint8')
            rows, _ = comb_matrix.shape
            vector_matrix = np.zeros((rows, lattice_size), dtype=np.uint8)
            for row in range(rows):
                np.add.at(vector_matrix[row], elements[row], 1)
            # end for
            elements = []
            feasible_row_indices = np.where((vector_matrix <= subset_size_list).all(axis=1))[0]
            vector_matrix_processed = vector_matrix[feasible_row_indices, :].transpose()
            res_matrix = processed_matrix.dot(vector_matrix_processed)
            feasible_row_indices = np.where((res_matrix.transpose() >= requirements).all(axis=1))[0]
            if len(feasible_row_indices) > 0:
                feasible_solution = vector_matrix_processed.transpose()[feasible_row_indices[0]]
                result = feasible_solution
                break;
            # end if
        if el is None:
            break;
        # end if
    # end while

    logfile.write("get-any-feasible-solution: [total:%.2fs]\n" % (time.time() - tstart))
    return result
