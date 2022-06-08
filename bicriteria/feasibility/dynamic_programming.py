import time

import numpy as np

from feasibility.bin_dec import int_to_k_array, k_array_to_int


def get_dynamic_programming_solution(processed_matrix,
                                     num_colors,
                                     num_of_medians,
                                     subset_size_list,
                                     requirement_list,
                                     logfile):
    tstart = time.time()
    for r in requirement_list:
        if r > num_of_medians:
            return None
        # end if
    # end for

    # preprocessing
    time_buf = time.time()
    converted_subset_size_list = list(map(lambda subset: min(num_of_medians, subset), subset_size_list))
    E_ = np.zeros((sum(converted_subset_size_list), num_colors), dtype=np.int8)
    processed_matrix_transposed = processed_matrix.transpose()
    i = 0
    for idx, size in enumerate(converted_subset_size_list):
        for _ in range(size):
            E_[i] = processed_matrix_transposed[idx]
            i = i + 1
        # end for
    # end for

    # allocate space for the dynamic programming array
    nof_rows = sum(converted_subset_size_list) + 1
    nof_colums = pow(num_of_medians + 1, num_colors)
    A = np.empty((nof_rows, nof_colums), dtype=np.int8)
    B = np.negative(np.ones((2, nof_colums, num_of_medians), dtype=np.int32))

    preproc_time = time.time() - time_buf
    time_buf = time.time()
    # dynamic programming recursion
    A[0].fill(num_of_medians + 1)
    A[0, 0] = 0
    # colums greater than the req to be stripped
    allowed_columns = np.zeros(nof_colums, dtype=bool)
    for j in range(nof_colums):
        current_set = int_to_k_array(j, num_of_medians, num_colors)
        diff = requirement_list - current_set
        allowed_columns[j] = np.all((diff >= 0))
    # end for

    for i in range(1, nof_rows):
        for j in range(nof_colums):
            if not allowed_columns[j]:
                continue
            # end if
            first_candidate = A[(i - 1) % 2, j]
            current_set = int_to_k_array(j, num_of_medians, num_colors)
            previous_set = current_set - E_[i - 1]

            previous_set[previous_set < 0] = 0
            previous_set_id = k_array_to_int(previous_set, num_of_medians)
            second_candidate = A[(i - 1) % 2, previous_set_id]
            new_value = min(first_candidate, second_candidate + 1, num_of_medians + 1)
            if new_value != A[i % 2, j]:
                A[i % 2, j] = new_value
                if new_value == first_candidate:
                    B[i % 2, j, :] = B[(i - 1) % 2, j, :]
                else:
                    B[i % 2, j, :] = B[(i - 1) % 2, previous_set_id, :]
                    for l in range(num_of_medians):
                        if B[i % 2, j, l] == -1:
                            B[i % 2, j, l] = i
                            break

        # end for
    # end for
    dynamic_time = time.time() - time_buf
    time_buf = time.time()

    req_id = k_array_to_int(requirement_list, num_of_medians)
    min_k = A[(nof_rows - 1) % 2, req_id]
    result = None
    if min_k > num_of_medians:
        return None
    # end if

    # get one solution
    # result = np.empty((min_k, num_colors), dtype=np.int8)
    result = list(map(lambda e: list(E_[e - 1]), B[(nof_rows - 1) % 2, req_id]))

    # end while
    result_time = time.time() - time_buf

    logfile.write("dynamic-program: [preproc: %.2fs, dynamic: %.2fs, result: %.2fs]" % \
                  (preproc_time, dynamic_time, result_time))
    total_time = time.time() - tstart
    logfile.write(" [total-time: %.2fs]\n" % (total_time))
    logfile.flush()

    return result
