import random
import sys
import time

from sklearn.datasets import make_blobs

import generator.get_feasible_instance
from es_fpt.es_fpt_3apx import es_fpt_3apx_complete


def test_es_fpt_complete():
    # random number generator seeds
    gen_seed = 12312321
    dist_matrix_seed = random.randint(1, int(pow(2, 32) - 1))
    local_search_seed = random.randint(1, int(pow(2, 32) - 1))

    # initialize
    logfile = sys.stdout
    n = 100
    t = 3
    k = 3
    d = 2
    r_max = 3
    r_min = 1
    max_freq = 3

    data, _ = make_blobs(n_samples=n, centers=k, n_features=d,
                         random_state=12312, cluster_std=0.8)

    # generate instance and time it
    time_buf = time.time()
    color_mat, rvec, _ = generator.get_feasible_instance.get_feasible_instance(
        t,
        n,
        r_max,
        r_min,
        max_freq,
        k,
        gen_seed,
        unique=False)
    es_fpt_3apx_complete(data, color_mat, rvec, k, logfile)


# end es_fpt_3apx_complete_test()


if __name__ == '__main__':
    test_es_fpt_complete()
