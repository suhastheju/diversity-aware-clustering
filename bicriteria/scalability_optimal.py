import random
from pprint import pprint

from scalability_optimal_internal import scalability_optimal

def scaling_nof_facilities_optimal(range_data, \
                                   objective, \
                                   results_dir, \
                                   strategy='local_search', \
                                   test_run=False):
    print("scaling_nof_facilities_optimal", range_data, strategy)
    init_seed = 123456789
    random.seed(init_seed)

    nof_facilities = range_data
    # nof_facilities  = list(range(1000, 20001, 1000))
    nof_centers = [3]
    nof_swaps = [1]
    nof_iterations = 2 if test_run else 5
    r_max = None
    return_solution = False

    logfile_name = "./%s/scaling_nof_facilities-%s.log" % \
                   (results_dir, strategy)
    logfile = open(logfile_name, 'w')

    result_file_name = "./%s/scaling_nof_facilities-%s.results" % \
                       (results_dir, strategy)
    result_file = open(result_file_name, 'w')

    stats = scalability_optimal(nof_facilities, \
                                nof_centers, \
                                nof_swaps, \
                                nof_iterations, \
                                objective, \
                                logfile, \
                                result_file, \
                                strategy)
    pprint(stats)


def scaling_nof_centers_optimal(range_data, \
                                objective, \
                                results_dir, \
                                strategy='local_search', \
                                test_run=False):
    print("scaling_nof_centers_optimal", range_data, strategy)
    init_seed = 123456789
    random.seed(init_seed)

    nof_facilities = [10000]
    nof_centers = range_data
    nof_swaps = [1]
    nof_iterations = 2 if test_run else 5
    return_solution = False

    logfile_name = "./%s/scaling_nof_centers-%s.log" % \
                   (results_dir, strategy)
    logfile = open(logfile_name, 'w')

    result_file_name = "./%s/scaling_nof_centers-%s.results" % \
                       (results_dir, strategy)
    result_file = open(result_file_name, 'w')

    stats = scalability_optimal(nof_facilities, \
                                nof_centers, \
                                nof_swaps, \
                                nof_iterations, \
                                objective, \
                                logfile, \
                                result_file, \
                                strategy)
    pprint(stats)


def test_batch_scaling_optimal(results_dir, test_run):
    for strategy in ['local_search_2', 'kmedoid']:
        if test_run:
            range_data = [10, 30, 100]
        else:
            range_data = [10, 100, 1000, 10000, 100000]
        scaling_nof_facilities_optimal(range_data, 'kmedian' if strategy == 'local_search_2' else 'kmeans', results_dir, strategy, test_run)

        if test_run:
            range_data = [1, 2, 3]
        else:
            range_data = [4, 5, 6, 7, 8, 9, 10, 11, 12]

        scaling_nof_centers_optimal(range_data, 'kmedian' if strategy == 'local_search_2' else 'kmeans', results_dir, strategy, test_run)

    # range_data = [1, 2, 3]
    # LS_scaling_nof_swaps(range_data, objective, results_dir)
