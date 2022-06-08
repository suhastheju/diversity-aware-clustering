import time

import numpy as np
import pandas as pd


def get_subset_map_and_size_list(color_matrix, logfile):
    tstart = time.time()
    t, N = color_matrix.shape
    for i in range(t):
        color_matrix[i] = color_matrix[i] * pow(2, t - i - 1)
    # end for
    records_array = np.sum(color_matrix, axis=0)
    subset_idx, subset_size_list = np.unique(records_array, return_counts=True)
    subset_map = pd.DataFrame(records_array).groupby([0]).indices

    logfile.write("get-subset-map-and-size-list: [total-subsets: %s, num-subsets: %d]" % \
                  (pow(2, color_matrix.shape[0]), len(subset_map.keys())))
    logfile.write(' [total: %.2fs]\n' % (time.time() - tstart))
    logfile.flush()

    return subset_map, subset_size_list
