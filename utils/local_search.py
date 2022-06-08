import time

import numpy as np
import psutil
from scipy.spatial.distance import cdist


def kmedian_local_search(data, K, init_seed=123456789, e=0.0):
    # data: N X d numpy array
    # K: number of clusters
    # init_seed: seed for psuedo-random number generator
    tstart = time.time()
    process = psutil.Process()

    data = data.astype(np.single)
    N, d = data.shape
    np.random.seed(init_seed)
    cur_medians_ = np.random.choice(data.shape[0], K, replace=False)
    cur_medians = data[cur_medians_]
    cur_cost = np.sum(cdist(data, np.array(cur_medians)).min(axis=1)) / N
    converged = False
    times = []
    S_star = cur_medians
    S_star_ = cur_medians_
    B_pos = np.zeros(shape=(1, d), dtype=np.single)

    iterations = 0
    while not converged:
        pre_cost = cur_cost
        for pos in range(K):
            start = time.time()
            # Remove for local-search swap
            B = S_star.copy()
            B[pos] = np.ones(d)
            temp_min = cdist(data, np.array(B)).min(axis=1)

            costs = []
            F_idx = []
            for idx in range(N):
                B_pos[0] = data[idx, :]
                cost = np.sum(np.minimum(cdist(data, B_pos).T, temp_min)) / N
                costs.append(cost)
                F_idx.append(idx)
            # end if
            if np.min(costs) < cur_cost:
                cur_cost = np.min(costs)
                idx = np.argmin(costs)
                S_star[pos, :] = data[idx, :]
                S_star_[pos] = idx
            # end for
            times.append((time.time() - start))
        # end for
        converged = (np.multiply(cur_cost, 1 + e) >= pre_cost)
        iterations += 1
    # end while

    stats = {}
    stats['facilities_indices'] = S_star_
    stats['cost'] = cur_cost
    stats['iterations'] = iterations
    stats['runtimes'] = times
    stats['peak_memory'] = process.memory_info().rss / (1024 * 1024)
    stats['virtual_memory'] = process.memory_info().vms / (1024 * 1024)
    stats['total_time'] = time.time() - tstart
    return stats
# end kMedian_local_search()
