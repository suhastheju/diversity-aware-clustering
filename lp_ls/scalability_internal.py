import itertools
import random
import sys
import time
from datetime import datetime
from pprint import pprint

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

import generator.get_feasible_instance
from lp_ls.lp_local_search import lp_ls_complete


def parse_result(result):
    return result.astype({'n': 'int32', \
                          'k': 'int32', \
                          't': 'int32', \
                          'lp_time': 'float64', \
                          'ls_time': 'float64', \
                          'total_time': 'float64', \
                          'peak_memory': 'float64', \
                          'virtual_memory': 'float64', \
                          'cost': 'float64',
                          'opt_cost': 'float64'})


def scalability(nof_facilities, \
                nof_centers, \
                nof_iterations, \
                nof_groups, \
                objective, \
                command, \
                logfile, \
                result_file, \
                init_seed=123456789):
    # logfile.write("%6s %3s %3s %5s %5s %8s %19s %9s\n"%\
    #              ("n", "t", "k", "r_min", "r_max",
    #               "max_freq", "seed", "inst_time"))
    # logfile.write("-------------------------------------------------------------\n")
    stats = pd.DataFrame(columns=['n', 'k', 't', \
                                  'lp_time', 'ls_time', \
                                  'total_time', \
                                  'peak_memory', \
                                  'virtual_memory', \
                                  'cost', 'opt_cost'], \
                         dtype='int32')
    np.random.seed(init_seed)
    for n, k, t in itertools.product(nof_facilities, nof_centers, nof_groups):
        for i in range(nof_iterations):
            tstart = time.time()
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print("Current Time =", current_time)
            sys.stdout.write('%8d %3d %3d %3d' % (n, k, t, i))

            data_seed = random.randint(1, int(pow(2, 32) - 1))
            gen_seed = random.randint(1, int(pow(2, 32) - 1))

            d = 5
            r_max = 3
            r_min = 1
            max_freq = t

            data, _ = make_blobs(n_samples=n, centers=k, n_features=d,
                                 random_state=data_seed, cluster_std=0.8)

            color_mat, rvec, _ = generator.get_feasible_instance.get_feasible_instance(
                t,
                n,
                r_max,
                r_min,
                max_freq,
                k,
                seed=gen_seed,
                unique=False)
            perf_stats = lp_ls_complete(data, color_mat, rvec, k, logfile)

            pprint(perf_stats, stream=logfile)

            stats.loc[len(stats)] = [n, k, t,
                                     perf_stats["lp_time"],
                                     perf_stats["ls_time"],
                                     perf_stats["total_time"],
                                     perf_stats['peak_memory'],
                                     perf_stats['virtual_memory'],
                                     perf_stats['cost'],
                                     perf_stats['opt_ls_cost']
                                     ]

            result_file.seek(0)
            result_file.truncate()
            # change datatype of columns
            result = parse_result(stats)
            result_file.write(result.to_string())
            result_file.write("\n----\n")
            result_file.write(result.to_json(orient='records'))
            result_file.flush()

            sys.stdout.write(' %.2fs\n' % (time.time() - tstart))
        # end for
    # end for
    return stats
