import sys
import time

import numpy as np
import psutil

from feasibility.bin_dec import bin_to_int
from feasibility.dynamic_programming import get_dynamic_programming_solution
from feasibility.generate_vector_matrix import generate_vector_matrix
from feasibility.get_any_feasible_solution import get_any_feasible_solution
from feasibility.get_feasible_solutions import get_feasible_solutions
from feasibility.get_processed_matrix import get_processed_matrix
from feasibility.get_subset_index_to_id import get_subset_index_to_id
from feasibility.get_subset_map_and_size_list import get_subset_map_and_size_list
from feasibility.linear_programming import get_linear_programming_solution
from feasibility.debug_details import debug_details


def feasible_solution_to_facilities(feasible_solution, subset_map):
    subset_index_to_id = get_subset_index_to_id(subset_map)
    facilities = []
    for idx, e in enumerate(feasible_solution):
        candidates = subset_map[subset_index_to_id[idx]]
        for i in range(e):
            facilities.append(candidates[i])
        # end for
    # end for
    return facilities


# end feasible_solution_to_facilities()


###############################################################################
def calculate(num_of_medians, \
              requirement_list, \
              color_matrix, \
              command, \
              return_solution=False, \
              logfile=sys.stdout,
              many_solution_lp_only=False):
    tstart = time.time()
    process = psutil.Process()
    tstart_temp = time.time()

    ## generate subset map
    num_colors = len(requirement_list)
    subset_map, subset_size_list = get_subset_map_and_size_list(color_matrix, \
                                                                logfile)

    input_time = time.time() - tstart_temp
    tstart_temp = time.time()

    ## generate matrix based on subset lattice
    use_sparse_matrix = False
    processed_matrix = get_processed_matrix(subset_map, num_colors,
                                            use_sparse_matrix, logfile)
    processed_time = time.time() - tstart_temp
    tstart_temp = time.time()
    vector_matrix = np.zeros(0)
    solution = None

    vector_time = 0.0
    if command == 'linear-program':
        tstart_temp = time.time()
        feasible_solution = get_linear_programming_solution(
            subset_size_list,
            processed_matrix,
            requirement_list,
            num_of_medians,
            many_solution_lp_only=many_solution_lp_only)
        nof_solutions = 0 if feasible_solution is None else 1
        if many_solution_lp_only:
            solution = feasible_solution
        if return_solution:
            if nof_solutions:
                solution = feasible_solution_to_facilities(feasible_solution, subset_map)
            else:
                solution = []
            # end if
        # end if
    elif command == 'dynamic-program':
        tstart_temp = time.time()
        # get a feasible solution
        feasible_solution = get_dynamic_programming_solution(
            processed_matrix,
            num_colors,
            num_of_medians,
            subset_size_list,
            requirement_list,
            logfile)
        nof_solutions = 0 if feasible_solution is None else 1
        if return_solution:
            facilities = []
            for row in feasible_solution:
                group_id = bin_to_int(np.flip(row))
                candidates = subset_map[group_id]
                facilities.append(candidates[0])
            # end for
            solution = facilities
        # end if
    elif command == 'brute-force':
        batch_size = 100
        tstart_temp = time.time()
        feasible_solution = get_any_feasible_solution(processed_matrix, \
                                                      requirement_list, \
                                                      subset_size_list, \
                                                      num_of_medians, \
                                                      batch_size, \
                                                      logfile)
        nof_solutions = 0 if feasible_solution is None else 1
        if return_solution:
            solution = feasible_solution_to_facilities(feasible_solution, subset_map)
        # end if
    else:
        ## generate vectors using intersection subset lattice
        vector_matrix = generate_vector_matrix(subset_size_list, \
                                               num_of_medians, \
                                               logfile)
        vector_time = time.time() - tstart_temp
        tstart_temp = time.time()
        feasible_solutions = get_feasible_solutions(processed_matrix, \
                                                    vector_matrix, \
                                                    requirement_list, \
                                                    logfile)
        solution = feasible_solutions
        nof_solutions = feasible_solutions.shape[0]
    # end if

    feasible_time = time.time() - tstart_temp
    ## log runtime
    logfile.write("calculate: [input: %.2fs, processed: %.2fs, vector: %.2fs, feasible: %.2fs]" % \
                  (input_time, processed_time, vector_time, feasible_time))
    total_time = time.time() - tstart
    logfile.write(" [total-time: %.2fs]\n" % (total_time))

    ## log debug information
    debug_details(process, tstart, processed_matrix, vector_matrix, logfile)

    # update performance statistics
    perf_stats = {}
    perf_stats['input_time'] = input_time
    perf_stats['solution'] = solution
    perf_stats['subset_map'] = subset_map
    perf_stats['processed_time'] = processed_time
    perf_stats['vector_time'] = vector_time
    perf_stats['feasible_time'] = feasible_time
    perf_stats['total_time'] = total_time
    perf_stats['nof_solutions'] = nof_solutions
    perf_stats['proc_matrix_shape'] = processed_matrix.shape
    perf_stats['vector_matrix_shape'] = vector_matrix.shape
    perf_stats['peak_memory'] = process.memory_info().rss / (1024 * 1024)
    perf_stats['virtual_memory'] = process.memory_info().vms / (1024 * 1024)

    return perf_stats


