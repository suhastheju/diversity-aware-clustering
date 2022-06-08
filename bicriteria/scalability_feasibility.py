import random
from pprint import pprint

import numpy as np

from scalability_feasibility_internal import scalability


def scaling_nof_facilities(command, \
                           unique=True, \
                           range=[100, 1000, 10000, 100000, 1000000, 10000000], \
                           objective=None, \
                           results_dir='exp-results', \
                           test_run=False,
                           many_solutions_lp_only=True):
    print("scaling_nof_facilities", command, unique, range)
    init_seed = 123456789
    random.seed(init_seed)

    nof_facilities = range
    nof_groups = [7]
    nof_centers = [4]
    nof_iterations = 10
    r_max = None
    return_solution = False

    logfile_name = "./%s/scaling_nof_facilities-%s.log" % \
                   (results_dir, command)
    logfile = open(logfile_name, 'w')

    result_file_name = "./%s/scaling_nof_facilities-%s%s.results" % \
                       (results_dir, command, '-unique' if unique else '')
    result_file = open(result_file_name, 'w')

    stats = scalability(nof_facilities, \
                        nof_groups, \
                        nof_centers, \
                        nof_iterations, \
                        r_max, \
                        unique, \
                        command, \
                        logfile, \
                        result_file, \
                        objective,
                        many_solutions_lp_only)
    pprint(stats)


def scaling_nof_centers(command, \
                        unique=True, \
                        range=[4, 5, 6, 7, 8, 9], \
                        objective=None, \
                        results_dir='exp-results', \
                        test_run=False,
                        many_solutions_lp_only=True):
    print("scaling_nof_centers", command, unique, range)
    init_seed = 123456789
    random.seed(init_seed)

    nof_facilities = [10000]
    nof_groups = [6]
    # 320 sec with 8 in the worst case with BF, expect 1600 sec with 9, so the total time about 5 hours
    nof_centers = range
    nof_iterations = 10
    r_max = 3
    return_solution = False

    logfile_name = "./%s/scaling_nof_centers-%s%s.log" % \
                   (results_dir, command, '-unique' if unique else '')
    logfile = open(logfile_name, 'w')

    result_file_name = "./%s/scaling_nof_centers-%s%s.results" % \
                       (results_dir, command, '-unique' if unique else '')
    result_file = open(result_file_name, 'w')

    stats = scalability(nof_facilities, \
                        nof_groups, \
                        nof_centers, \
                        nof_iterations, \
                        r_max, \
                        unique, \
                        command, \
                        logfile, \
                        result_file, \
                        objective,
                        many_solutions_lp_only)

    pprint(stats)


def scaling_nof_groups(command, \
                       unique=True, \
                       range=[4, 5, 6, 7, 8], \
                       objective=None, \
                       results_dir='exp-results',
                       test_run=False,
                       many_solutions_lp_only=True):
    print("scaling_nof_groups", command, unique, range)
    init_seed = 123456789
    random.seed(init_seed)

    nof_facilities = [10000]
    nof_groups = range
    nof_centers = [5]
    nof_iterations = 10
    r_max = 3
    return_solution = False

    logfile_name = "./%s/scaling_nof_groups-%s%s.log" % \
                   (results_dir, command, '-unique' if unique else '')
    logfile = open(logfile_name, 'w')

    result_file_name = "./%s/scaling_nof_groups-%s%s.results" % \
                       (results_dir, command, '-unique' if unique else '')
    result_file = open(result_file_name, 'w')

    stats = scalability(nof_facilities, \
                        nof_groups, \
                        nof_centers, \
                        nof_iterations, \
                        r_max, \
                        unique, \
                        command, \
                        logfile, \
                        result_file, \
                        objective,
                        many_solutions_lp_only)
    pprint(stats)


def test_batch_scaling(objective, results_dir, test_run=False):
    for unique in [True, False]:  # worst case first
        for algo_type in ['linear-program', 'brute-force', 'dynamic-program']:
            ################################
            # scaling 'k'
            if test_run:
                range_data = list(range(4, 5))
            elif algo_type == 'linear-program':
                range_data = list(range(4, 30))
            elif algo_type == 'brute-force':
                if unique:
                    range_data = list(range(4, 10))
                else:
                    range_data = list(range(4, 30))
                # end if
            elif algo_type == 'dynamic-program':
                if unique:
                    range_data = list(range(4, 13))
                else:
                    range_data = list(range(4, 11))
                # end if
            # end if

            scaling_nof_centers(algo_type, \
                                unique, \
                                range_data, \
                                objective, \
                                results_dir, \
                                test_run)

            ################################
            # scaling 't'
            if test_run:
                range_data = list(range(4, 5))
            elif algo_type == 'linear-program':
                range_data = list(range(4, 15))
            elif algo_type == 'brute-force':
                if unique:
                    range_data = list(range(4, 9))  # 9 takes over 2 hours, predict 40 for tests
                else:
                    range_data = list(range(4, 12))
            elif algo_type == 'dynamic-program':
                range_data = list(range(4, 9))
            # end if
            scaling_nof_groups(algo_type, \
                               unique, \
                               range_data, \
                               objective, \
                               results_dir, \
                               test_run)

            ################################
            # scaling 'n'
            if test_run:
                range_data = [100]
            else:
                range_data = np.logspace(3, 9, num=9 - 3, endpoint=False).astype(int)  # use 10 in the final version
            # end if
            # override to test script
            scaling_nof_facilities(algo_type, \
                                   unique, \
                                   range_data, \
                                   objective, \
                                   results_dir, \
                                   test_run)
        # end for
    # end for
