import itertools
import random
import sys
import time
from datetime import datetime
from pprint import pprint

import pandas as pd

import generator.get_distance_matrix
import generator.get_feasible_instance
from feasibility import feasibility
from scalability_internal.parse_results import parse_result
from utils.local_search import kmedian_local_search


def scalability(nof_facilities, \
                nof_groups, \
                nof_centers, \
                nof_iterations, \
                given_r_max, \
                unique, \
                command, \
                logfile, \
                result_file, \
                objective,
                many_solutions_lp_only=False):
    stats = pd.DataFrame(columns=['n', 't', 'k', 'r_min', 'r_max', 'max_freq', \
                                  'seed', 'instance_time', 'feasibility_time', \
                                  'virtual_memory', 'peak_memory', \
                                  'objective_time', 'input_time_', 'processed_time_', 'vector_time_', 'feasible_time_', 'total_time_'], \
                         dtype='int32')
    for n, t, k in itertools.product(nof_facilities, nof_groups, nof_centers):
        for i in range(nof_iterations):
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print("Current Time =", current_time)
            print(n, t, k, i, command)

            # random number generator seeds
            gen_seed = random.randint(1, sys.maxsize)
            dist_matrix_seed = random.randint(1, int(pow(2, 32) - 1))
            local_search_seed = random.randint(1, int(pow(2, 32) - 1))

            # initialize
            r_max = given_r_max or min(t, k)
            r_min = 1
            max_freq = int(t / 2) + 1

            # generate instance and time it
            time_buf = time.time()
            color_mat, rvec, solution = generator.get_feasible_instance.get_feasible_instance(
                t,
                n,
                r_max,
                r_min,
                max_freq,
                k,
                gen_seed,
                unique)
            instance_time = time.time() - time_buf

            # find a feasible solution and time it
            time_buf = time.time()
            return_solution = False
            perf_stats = feasibility.calculate(k, rvec, color_mat,
                                               command, return_solution,
                                               logfile, many_solutions_lp_only)
            feasibility_time = time.time() - time_buf

            # find cluster centers based on objective and time it
            objective_time = 0
            dist_matrix_time = 0
            ls_stats = {}
            if objective != None:
                time_buf = time.time()
                dist_matrix = generator.get_distance_matrix.get_distance_matrix(n, dist_matrix_seed)
                dist_matrix_time = time.time() - time_buf

                time_buf = time.time()
                ls_stats = kmedian_local_search(dist_matrix,
                                                k,
                                                local_search_seed,
                                                0.0)
                objective_time = time.time() - time_buf
            # end if

            # printing stats
            pprint(perf_stats, stream=logfile)
            pprint(ls_stats, stream=logfile)

            peak_memory = perf_stats['peak_memory']
            virtual_memory = perf_stats['virtual_memory']
            virtual_memory = perf_stats['virtual_memory']
            input_time_ = perf_stats['input_time']
            processed_time_ = perf_stats['processed_time']
            vector_time_ = perf_stats['vector_time']
            feasible_time_ = perf_stats['feasible_time']
            total_time_ = perf_stats['total_time']
            logfile.write("%6d %3d %3d %5d %5d %8d %d" % \
                          (n, t, k, r_min, r_max, max_freq, gen_seed))
            logfile.write(" %9.4f %9.4f %9.4f %9.4f %9.4f %9.4f %9.4f %9.4f %9.4f" % \
                          (instance_time, feasibility_time, dist_matrix_time, objective_time, input_time_, processed_time_, vector_time_, feasible_time_, total_time_))
            logfile.write("\n======================================================================\n")
            logfile.write("\n\n\n")
            logfile.flush()

            # append results to pandas dataframe
            stats.loc[len(stats)] = [n, t, k, r_min, r_max, max_freq, \
                                     gen_seed, instance_time, feasibility_time, \
                                     virtual_memory, peak_memory, objective_time, \
                                     input_time_, processed_time_, vector_time_, feasible_time_, total_time_]
            result_file.seek(0)
            result_file.truncate()
            # change datatype of columns
            result = parse_result(stats)
            result_file.write(result.to_string())
            result_file.write("\n----\n")
            result_file.write(result.to_json(orient='records'))
            result_file.flush()
        # end for
    # end for

    # change datatype of columns
    return parse_result(stats)
