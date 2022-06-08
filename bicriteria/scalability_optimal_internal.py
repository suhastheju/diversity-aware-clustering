import itertools
import random
import time
from datetime import datetime
from pprint import pprint

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

import generator.get_distance_matrix
from scalability_internal.parse_stats import parse_stats_LS
from utils import local_search
from utils.kmeans import kmedoids_sklearn
from utils.local_search import kmedian_local_search

d = 2


def scalability_optimal(nof_facilities, \
                        nof_centers, \
                        nof_swaps, \
                        nof_iterations, \
                        objective, \
                        logfile, \
                        result_file, \
                        strategy='local_search_2'):
    stats = pd.DataFrame(columns=['n', 'k', 'swaps', 'strategy', 'objective', \
                                  'dist_matrix_seed', 'local_search_seed', \
                                  'dist_matrix_time', 'objective_time', \
                                  'virtual_memory', 'peak_memory'], \
                         dtype='int32')
    for n, k, swaps in itertools.product(nof_facilities, nof_centers, nof_swaps if strategy == 'local_search' else [0]):
        for i in range(nof_iterations):
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print("Current Time =", current_time)
            print(n, k, i, objective, strategy)

            # random number generator seeds
            dist_matrix_seed = random.randint(1, int(pow(2, 32) - 1))
            local_search_seed = random.randint(1, int(pow(2, 32) - 1))

            # find cluster centers based on objective and time it
            computation_stats = {}
            time_buf = time.time()
            if strategy == 'local_search':
                dist_matrix = generator.get_distance_matrix.get_distance_matrix(n, dist_matrix_seed)
                dist_matrix_time = time.time() - time_buf

                time_buf = time.time()
                computation_stats = local_search.local_search(dist_matrix,
                                                              dist_matrix,
                                                              k,
                                                              objective,
                                                              local_search_seed,
                                                              swaps,
                                                              logfile)
            elif strategy == 'kmeans_mlpack':
                data = np.random.uniform(low=0.0, high=1.0, size=(n, d))
                dist_matrix_time = time.time() - time_buf
                time_buf = time.time()
                computation_stats = kmeans_mlpack(data, k)
            elif strategy == 'kmeans_sklearn':
                data = np.random.uniform(low=0.0, high=1.0, size=(n, d))
                dist_matrix_time = time.time() - time_buf
                time_buf = time.time()
                computation_stats = kmeans_sklearn(data, k)
            elif strategy == 'local_search_2':
                data, _ = make_blobs(n_samples=n,
                                     centers=k,
                                     n_features=d,
                                     random_state=0,
                                     cluster_std=0.8)
                dist_matrix_time = time.time() - time_buf
                time_buf = time.time()
                computation_stats = kmedian_local_search(data, k)
            elif strategy == 'kmedoid':
                data = np.random.uniform(low=0.0, high=1.0, size=(n, d))
                dist_matrix_time = time.time() - time_buf
                time_buf = time.time()
                computation_stats = kmedoids_sklearn(data, k)
            # end if
            objective_time = time.time() - time_buf

            # printing stats
            pprint(computation_stats, stream=logfile)

            peak_memory = computation_stats['peak_memory']
            virtual_memory = computation_stats['virtual_memory']
            logfile.write("%6d %3d %3d %d %d" % \
                          (n, k, swaps, dist_matrix_seed, local_search_seed))
            logfile.write(" %9.4f %9.4f %.2f %.2f" % \
                          (dist_matrix_time, objective_time, virtual_memory, peak_memory))
            logfile.write("\n======================================================================\n")
            logfile.write("\n\n\n")
            logfile.flush()

            # append results to pandas dataframe
            stats.loc[len(stats)] = [n, k, swaps, strategy, objective, \
                                     dist_matrix_seed, local_search_seed, \
                                     dist_matrix_time, objective_time, \
                                     virtual_memory, peak_memory]
            result_file.seek(0)
            result_file.truncate()
            # change datatype of columns
            result = parse_stats_LS(stats)
            result_file.write(result.to_string())
            result_file.write("\n----\n")
            result_file.write(result.to_json(orient='records'))
        # end for
    # end for

    # change datatype of columns
    return parse_stats_LS(stats)
