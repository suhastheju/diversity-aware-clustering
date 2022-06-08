import sys
import time

import numpy as np
import psutil

from bicriteria import feasibility
from utils.kmedkpm import k_median_k_partitions_LS
from utils.local_search import kmedian_local_search

test = False


def lp_ls_complete(data, color_mat, rvec, k, logfile):
    ############################################################################
    # INPUT
    # data: N X d numpy array
    # color_mat: N*t numpy array representing groups' memberships
    # rvec: requirements vector of t size
    # k: number of clusters

    # OUTPUT
    # Object with stats, i.e, with "cost" that is a cost of the solution
    ############################################################################
    (N, d) = data.shape
    many_solutions_lp_only = True
    command = 'linear-program'
    return_solution = False
    process = psutil.Process()
    tstart = time.time()
    time_buf = time.time()

    perf_stats = feasibility.calculate(k, rvec, color_mat,
                                       command, return_solution,
                                       logfile, many_solutions_lp_only)

    set_mappings = perf_stats["subset_map"]
    solution = perf_stats["solution"]
    set_to_indices = {}

    for (idx, _id) in enumerate(sorted(set_mappings.keys())):
        set_to_indices[idx] = _id
    # end for

    unique_solutions = solution if len(solution) == 0 else np.unique(np.stack(solution, axis=0), axis=0)
    print('solutions: ', 0 if len(solution) == 0 else unique_solutions.shape[0])
    total_cost = sys.maxsize
    time_buf = time.time()
    for (_, s) in enumerate(unique_solutions):
        E = {}
        i = 0
        for (idx, e) in enumerate(s):
            for _ in range(e):
                # E[i] = (idx, set_to_indices[idx])
                E[i] = data[set_mappings[set_to_indices[idx]], :]
                i = i + 1
            # end for
        # end for
        if k > i:
            continue

        statc = k_median_k_partitions_LS(E, data, None, N, d, k, is_coreset=False)
        total_cost = min(total_cost, statc["cost"])
    # print(set_to_indices)
    kmedkpmtime = time.time() - time_buf
    total_time = time.time() - tstart

    stats_total = {}
    opt_ls_cost = kmedian_local_search(data, k)["cost"]
    stats_total['opt_ls_cost'] = opt_ls_cost
    stats_total["lp_time"] = perf_stats["total_time"]
    stats_total["total_time"] = total_time
    stats_total["ls_time"] = kmedkpmtime
    stats_total['peak_memory'] = process.memory_info().rss / (1024 * 1024)
    stats_total['virtual_memory'] = process.memory_info().vms / (1024 * 1024)
    stats_total['cost'] = total_cost
    return stats_total

# end lp_ls_complete()
