import math
import sys
import time

import numpy as np
import psutil
from scipy.spatial.distance import cdist

# local imports
from feasibility.feasibility import calculate
from utils.kmeans import kmedoids_sklearn
from utils.local_search import kmedian_local_search

dist_matrix_seed = 123123

test = False


def parse_result(result):
    return result.astype({'n': 'int32', \
                          't': 'int32', \
                          'k': 'int32', \
                          'total_time': 'float64', \
                          'feasible_solution_processed_time': 'float64', \
                          'feasible_solution_vector_time': 'float64', \
                          'feasible_solution_total_time': 'float64', \
                          'optimal_solution_time': 'float64', \
                          'peak_memory': 'float64', \
                          'virtual_memory': 'float64', \
                          'solution_cost': 'float64', \
                          'opt_cost': 'float64', \
                          'solution_size': 'int32'})


# end parse_result()

def cost(data, indices, objective='kmedian'):
    N, d = data.shape
    S = data[indices, :]
    cost = math.inf
    if objective == 'kmedian':
        cost = cdist(data, S).min(axis=1).sum() / N
    elif objective == 'kmeans':
        cost = np.square(cdist(data, S).min(axis=1)).sum() / N
    return cost


# end cost()


def bicriteria(data, \
               k, \
               color_mat, \
               rvec, \
               objective='kmedian', \
               strategy='brute-force', \
               logfile=sys.stdout):
    ############################################################################
    # INPUT
    # data: N X d numpy array
    # K: number of clusters
    # color_mat: N*t numpy array representing groups' memberships
    # rvec: requirements vector of t size
    # strategy: 'linear-program', 'dynamic-program' or 'brute-force' (exhaustive search)

    # OUTPUT
    # Object with stats, i.e, with "result" that is a list of indices of selected facilities
    ############################################################################

    process = psutil.Process()
    t = len(rvec)
    color_mat_ = color_mat
    rvec_ = np.copy(rvec)

    tstart = time.time()

    if objective == 'kmeans':
        computation_stats_1 = kmedoids_sklearn(data, k)
        facilities_indices = computation_stats_1["cluster_centers"]
    elif objective == 'kmedian':
        computation_stats_1 = kmedian_local_search(data, k)
        facilities_indices = computation_stats_1["facilities_indices"]
    else:
        return
    # end if

    for i in facilities_indices:
        for g in range(t):
            # groups are in opposite order in color_mat
            if color_mat_[t - g - 1, i] == 1:
                rvec_[g] = max(rvec_[g] - 1, 0)
                # zeroizong not to select again
                color_mat_[t - g - 1, i] = 0
            # end if
        # end for
    # end for

    computation_stats_2 = calculate(k, rvec_, color_mat_, strategy, True, logfile)
    if computation_stats_2 != None:
        result = list(set([*computation_stats_2["solution"], *facilities_indices]))
    else:
        result = []
    # end if

    total_time = time.time() - tstart
    feasible_solution_processed_time = computation_stats_2['processed_time']
    feasible_solution_vector_time = computation_stats_2['vector_time']
    feasible_solution_feasible_time = computation_stats_2['feasible_time']
    feasible_solution_total_time = computation_stats_2['total_time']
    optimal_solution_time = computation_stats_1["total_time"]
    peak_memory = process.memory_info().rss / (1024 * 1024)
    virtual_memory = process.memory_info().vms / (1024 * 1024)

    perf_stats = {}
    perf_stats["total_time"] = total_time
    perf_stats["feasible_solution_processed_time"] = feasible_solution_processed_time
    perf_stats["feasible_solution_vector_time"] = feasible_solution_vector_time
    perf_stats["feasible_solution_feasible_time"] = feasible_solution_feasible_time
    perf_stats["feasible_solution_total_time"] = feasible_solution_total_time
    perf_stats["optimal_solution_time"] = optimal_solution_time
    perf_stats["peak_memory"] = peak_memory
    perf_stats["virtual_memory"] = virtual_memory
    perf_stats["result"] = result
    perf_stats["solution_cost"] = cost(data, result, objective)
    perf_stats["solution_size"] = len(result)
    perf_stats["opt_cost"] = cost(data, facilities_indices, objective)
    return perf_stats
# end bicriteria()
