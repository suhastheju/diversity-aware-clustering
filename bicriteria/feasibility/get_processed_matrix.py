import time

import numpy as np
from scipy import sparse

from feasibility.bin_dec import int_to_bin


def get_processed_matrix(subset_map, t, use_sparse, logfile):
    tstart = time.time()
    processed_matrix = np.empty((0, t), np.int8)
    for key in sorted(subset_map.keys()):
        bit_vec = int_to_bin(key, t)
        processed_matrix = np.append(processed_matrix, [bit_vec], axis=0)
    # end for
    logfile.write("get-processed-matrix: [total: %.2fs]\n" % (time.time() - tstart))
    logfile.flush()

    if use_sparse:
        return sparse.csr_matrix(processed_matrix.transpose())
    return processed_matrix.transpose()
