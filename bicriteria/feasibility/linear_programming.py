from math import floor
from random import random

import numpy as np
from scipy.optimize import linprog


###############################################################################
# linear program implementation
# The function works in two ways:
# 1) many_solution_lp_only = False
# returns a single feasible solution and make a few retries
# randomizng the objecttive function and reruning randomized raouding
# 2) many_solution_lp_only = True
# Here the program does not stop after the first success
# but returns as many solutions as possible within limited retrials
# This is used by LP_LS heuristic.

def get_linear_programming_solution(subset_size_list, \
                                    processed_matrix, \
                                    requirement_list, \
                                    num_of_medians, \
                                    many_solution_lp_only=False):
    v = np.ones(subset_size_list.size, dtype=np.uint8)
    if many_solution_lp_only:
        A = np.multiply(processed_matrix, -1)
        B = np.multiply(requirement_list, -1)
    else:
        A = np.append(np.multiply(processed_matrix, -1), [v], axis=0)
        B = np.append(np.multiply(requirement_list, -1), num_of_medians)
    number_of_retries_1 = 50 if many_solution_lp_only else 5
    number_of_retries_2 = 2 if many_solution_lp_only else 3
    many_solutions = []
    bounds = list(map(lambda x: (0, x), subset_size_list))
    for _ in range(number_of_retries_1):
        f = np.random.rand(subset_size_list.size)
        if many_solution_lp_only:
            result = linprog(f, A, B, A_eq=[v], b_eq=[num_of_medians], bounds=bounds)
        else:
            result = linprog(f, A, B, bounds=bounds)
        if not result.success:
            break
        # end if
        for __ in range(number_of_retries_2):
            rounded_result = np.array(list(map(lambda x: floor(x + random()), result.x)))
            if (rounded_result <= subset_size_list).all(axis=0):
                res_matrix = processed_matrix.dot(rounded_result.reshape(-1, 1))
                if (res_matrix.transpose() >= requirement_list).all(axis=1)[0]:
                    if many_solution_lp_only:
                        many_solutions.append(rounded_result)
                    else:
                        return rounded_result
                # end if
            # end if
        # end for
    # end for
    return many_solutions if many_solution_lp_only else None
