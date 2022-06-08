import time

import numpy as np
import pandas as pd
from pyclustering.cluster.kmedians import kmedians
from scipy.spatial.distance import cdist


def kmedian_coreset(data, K, coreset_size=0.1, init_seed=123456789):
    ############################################################################
    # INPUT
    # data: N X d numpy array
    # K: number of clusters
    # coreset_size: 0.0 < value <= 1.0, size of the coreset
    # init_seed: seed for initializing psuedo-random number generator

    # OUTPUT
    # C_idx: index of coreset
    # C_s: chosen coreset data points
    # C_w: weight of coreset points
    ############################################################################

    tstart = time.time()
    assert coreset_size > 0.0 and coreset_size <= 1.0;

    # Initialize
    N, d = data.shape
    nof_trials = d * d  # d^2 iterations
    np.random.seed(init_seed)

    # compute KMedian score
    init_medians = data[np.random.choice(data.shape[0], K, replace=False)]
    km_model = kmedians(data, init_medians)
    km_model.process()
    medians = km_model.get_medians()
    cost_QX = np.sum(cdist(data, np.array(medians)).min(axis=1)) / N

    cost_QB = np.inf
    for trial in range(nof_trials):
        np.random.seed(trial)
        fst_idx = np.random.choice(N)  # Choosing first center randomly
        B = []  # Approximate cluster centers
        B.append(data[fst_idx])
        for choice in range(K - 1):  # Choice of remaining K-1 centers
            proba = cdist(data, np.array(B)).min(axis=1)
            norm_proba = proba / np.sum(proba)
            idx = np.random.choice(N, p=norm_proba)
            B.append(data[idx, :])
        # end for

        tmp_cost = cdist(data, np.array(B)).min(axis=1).sum() / N
        if tmp_cost < cost_QB:
            cost_QB = tmp_cost
            B_star = B.copy()
    # end for

    alpha = 16 * (np.log2(K) + 2)
    alpha_R = cost_QB / cost_QX;
    Q_B = np.array(B_star)

    B_y = cdist(data, Q_B).argmin(axis=1);  # Cluster labels according to Q_B
    cost_QB = cdist(data, Q_B).min(axis=1);  # Cost(X, Q_B)
    median_cost_QB = np.median(cost_QB);
    cost_QB_cluster = [np.median(cost_QB[B_y == C]) for C in range(K)];  # Cost of each cluster
    cluster_n = pd.Series(B_y).sort_index().value_counts().values  # Cardinality of each cluster
    S = 6 * alpha + 4 * K;  # Total sensitivity

    # Sensitivity
    s_x = np.array([2 * alpha * cost_QB[i] / median_cost_QB + \
                    4 * alpha * cost_QB_cluster[B_y[i]] / median_cost_QB + \
                    4 * N / cluster_n[B_y[i]] for i in range(N)])
    # Probability
    q_x = s_x / S / N
    # print(np.sum(q_x).round(2))
    # assert np.sum(q_x).round(2) == 1.;
    q_x = q_x / q_x.sum()  # Adjusting for numerical precision to make sum(q)=1

    C = int(coreset_size * N)  # Number of points to draw (coreset size)
    C_idx = np.random.choice(N, size=C + K, p=q_x)  # Coreset index
    C_s = data[C_idx]  # Coreset points
    C_w = 1 / q_x[C_idx]  # Corresponding weights
    total_time = time.time() - tstart

    stats = {}
    stats['indexes'] = C_idx
    stats['coreset'] = C_s
    stats['weights'] = C_w
    stats['total_time'] = total_time
    return stats
