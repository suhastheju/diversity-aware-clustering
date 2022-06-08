import math
import time
from itertools import combinations_with_replacement
from itertools import product
from sys import stdout

import numpy as np
import psutil
import skgstat
from scipy.spatial.distance import cdist
from sklearn.datasets import make_blobs

from coresets.kmedian_coreset import kmedian_coreset


def print_statistics(logfile=stdout):
    process = psutil.Process()
    logfile.write("=======================================================================\n")
    logfile.write("STATISTICS: k-median-k-partitions-3eps\n")
    logfile.write("--------------\n")
    logfile.write("CPU:    [CORES-TOTAL: %s, THREADS-TOTAL: %s, THREADS-USED: %s]\n" % \
                  (psutil.cpu_count(logical=False), psutil.cpu_count(), process.num_threads()))
    logfile.write("\t       [CUR-FREQ: %f, MIN-FREQ: %f, MAX-FREQ: %f]\n" % \
                  (psutil.cpu_freq().current, psutil.cpu_freq().min, \
                   psutil.cpu_freq().max))
    logfile.write("MEMORY: [RAM-MEM: %.2fMB, VIR-MEM: %.2fMB, TOTAL: %.2fMB]\n" % \
                  (process.memory_info().rss / (1024 * 1024), \
                   process.memory_info().vms / (1024 * 1024), \
                   (process.memory_info().rss + process.memory_info().vms) / (1024 * 1024)))
    logfile.write("======================================================================\n")
    logfile.flush()


# end debug_details()


################################################################################
# binning implementation
def get_binned_distances(data, coreset, weights, k):
    N, d = data.shape
    distances = np.unique(cdist(data, coreset) * weights)
    maxlag = distances.max()
    nof_bins = int(k * math.log(N, 2))

    binned_distances, _ = skgstat.binning.kmeans(distances,
                                                 nof_bins,
                                                 maxlag)
    return binned_distances


# end get_binned_distances()


################################################################################
# implementation of (3 + \epsilon) approximation algorithm for
# k-median-k-partitions problem 
# for details see Algorithm~2 in Supplementary material
def k_median_k_partitions_3eps(E, \
                               coreset, \
                               weights, \
                               N, \
                               d, \
                               k, \
                               binned_distances, \
                               binned_index, \
                               logfile=stdout):
    tstart = time.time()
    S_star = []
    cost_star = math.inf
    c_j = np.zeros(shape=(1, d))
    n, _ = coreset.shape

    iterations = 0
    for C in combinations_with_replacement(range(n), k):
        for Lambda in combinations_with_replacement(binned_distances, k):
            S = []
            for j in range(0, k):
                idx = binned_index[Lambda[j]]
                Lambda_min = binned_distances[idx] if idx == 1 else binned_distances[idx - 1]
                Lambda_max = Lambda[j]
                c_j[0] = coreset[C[j]]
                w_j = weights[C[j]]

                dist = np.multiply(cdist(E[j], c_j).min(axis=1), w_j)
                indices = np.array(np.where(np.logical_and(dist >= Lambda_min, dist <= Lambda_max))).flatten()
                if np.size(indices) > 0:
                    Pi_j = E[j][indices, :]
                    S.append(Pi_j[0])
                # end if
            # end for

            # Compute cost of solution
            S_tmp = np.array(S)
            if np.size(S_tmp) > 0:
                cost = np.sum(np.multiply(cdist(coreset, np.array(S_tmp)).min(axis=1), weights)) / N
                if cost_star > cost:  # Update cost and solution
                    cost_star = cost
                    S_star = np.array(S_tmp)
                # end if
            # end if
            iterations += 1
        # end for
    # end for
    total_time = time.time() - tstart

    logfile.write("\nk-med-k-part-3eps: [iterations: %d] [time: %.2f s]\n" % \
                  (iterations, total_time))
    logfile.flush()
    return cost_star, S_star


# end k_median_k_partitions_3eps()


################################################################################
def test_k_median_k_partitions_3eps(N, d, k):
    # N=100
    # d=2
    # k=3
    data, _ = make_blobs(n_samples=N,
                         centers=k,
                         n_features=d,
                         random_state=0,
                         cluster_std=0.8)

    stats = kmedian_coreset(data, k, coreset_size=0.1)
    coreset = stats['coreset']
    weights = stats['weights']
    # coreset, _ = make_blobs(n_samples=int(N*0.1),
    #                 centers=k,
    #                 n_features=d,
    #                 random_state=0,
    #                 cluster_std=0.8)
    # weights = np.random.rand(int(N*0.1))

    binned_distances = get_binned_distances(data, coreset, weights, k)
    binned_index = dict(zip(binned_distances, range(len(binned_distances))))

    E = {}
    for j in range(k):
        indices = np.random.randint(N, size=10)
        E[j] = np.array(data[indices, :])
    # end for

    cost, S = k_median_k_partitions_3eps(E, coreset, weights, N, d, k, binned_distances, binned_index)
    print(cost, S)
    print_statistics()


# end test_k_median_k_partitions_3eps()

def test_scaling():
    NN = [100]
    dd = [4]
    kk = [3, 4, 5]

    for N, d, k in product(NN, dd, kk):
        tstart = time.time()
        test_k_median_k_partitions_3eps(N, d, k)
        total_time = time.time() - tstart
        print(N, d, k, total_time)
    # end for


# end test_scaling()
# test_scaling()

################################################################################
# implementation of local-search heuristic for k-median-k-partitions problem
# Algorithm description is available in Experimental setup section of the paper,
# which is denoted as LS_1.
def k_median_k_partitions_LS(E, \
                             coreset, \
                             weights, \
                             N, \
                             d, \
                             k, \
                             init_seed=123456789, \
                             epsilon=0.0, \
                             is_coreset=True):
    tstart = time.time()
    process = psutil.Process()

    coreset = coreset.astype(np.single)

    np.random.seed(init_seed)
    cur_medians_idx = np.zeros(k, dtype=int)
    cur_medians = np.zeros(shape=(k, d), dtype=np.single)
    for j in range(k):
        E_j = E[j]
        idx = np.random.choice(E_j.shape[0], 1, replace=False)
        cur_medians_idx[j] = idx
        cur_medians[j, :] = E_j[idx, :]
    # end for
    if is_coreset:
        cost_star = np.sum(np.multiply(cdist(coreset, np.array(cur_medians)).min(axis=1), weights)) / N
    else:
        cost_star = np.sum(cdist(coreset, np.array(cur_medians)).min(axis=1)) / N
    # end if

    converged = False
    times = []
    S_star = cur_medians
    S_star_idx = cur_medians_idx
    B_j = np.zeros(shape=(1, d), dtype=np.single)
    iterations = 0
    while not converged:
        pre_cost = cost_star
        for j in range(k):
            start = time.time()
            # Remove for local-search swap
            B = S_star.copy()
            B[j] = np.ones(d)
            temp_min = cdist(coreset, B).min(axis=1)

            costs = []
            F_idx = []
            E_j = E[j]
            for idx in range(E_j.shape[0]):
                B_j[0] = E_j[idx, :]
                if is_coreset:
                    cost = np.sum(np.multiply(np.minimum(cdist(coreset, B_j).T, temp_min), weights)) / N
                else:
                    cost = np.sum(np.minimum(cdist(coreset, B_j).T, temp_min)) / N
                # endif
                costs.append(cost)
                F_idx.append(idx)
            # end if

            if np.min(costs) < cost_star:
                cost_star = np.min(costs)
                idx = np.argmin(costs)
                S_star[j, :] = E_j[idx, :]
                S_star_idx[j] = idx
            # end for
            times.append((time.time() - start))
        # end for
        converged = (np.multiply(cost_star, 1 + epsilon) >= pre_cost)
        iterations += 1
    # end while

    stats = {}
    stats['facilities'] = S_star
    stats['facilities_indices'] = S_star_idx
    stats['cost'] = cost_star
    stats['iterations'] = iterations
    stats['runtimes'] = times
    stats['peak_memory'] = process.memory_info().rss / (1024 * 1024)
    stats['virtual_memory'] = process.memory_info().vms / (1024 * 1024)
    stats['total_time'] = time.time() - tstart
    return stats
# end k_median_k_partitions_LS()
