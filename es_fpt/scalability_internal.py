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
from es_fpt.es_fpt_3apx import es_fpt_3apx_complete


def parse_result(result):
    return result.astype({'n': 'int32', \
                          'k': 'int32', \
                          't': 'int32', \
                          'total_time': 'float64', \
                          'feasible_time': 'float64', \
                          'coreset_time': 'float64', \
                          'fpt_3apx_time': 'float64', \
                          'peak_memory': 'float64', \
                          'virtual_memory': 'float64', \
                          'cost': 'float64', \
                          'opt_cost': 'float64', \
                          'solution_size': 'int32', \
                          'nof_solutions': 'int32', \
                          'threshold': 'int32'})


def scalability(nof_facilities, \
                nof_centers, \
                nof_iterations, \
                nof_groups, \
                objective, \
                command, \
                logfile, \
                result_file, \
                init_seed=123456789):
    stats = pd.DataFrame(columns=['n', 'k', 't', \
                                  'total_time', 'feasible_time', \
                                  'coreset_time', \
                                  'fpt_3apx_time', \
                                  'peak_memory', \
                                  'virtual_memory', \
                                  'cost', 'opt_cost', 'solution_size', \
                                  'nof_solutions', \
                                  'threshold'], \
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

            d = 3
            r_max = 3
            r_min = 1
            max_freq = 3

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
            perf_stats = es_fpt_3apx_complete(data, color_mat, rvec, k, logfile)

            pprint(perf_stats, stream=logfile)

            stats.loc[len(stats)] = [n, k, t, perf_stats["total_time"], \
                                     perf_stats['feasible_time'], \
                                     perf_stats['coreset_time'], \
                                     perf_stats['fpt_3apx_time'], \
                                     perf_stats['peak_memory'], \
                                     perf_stats['virtual_memory'], \
                                     perf_stats['cost'], \
                                     perf_stats['opt_ls_cost'], \
                                     k,
                                     perf_stats['nof_solutions'], \
                                     perf_stats['threshold']]

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
