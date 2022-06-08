import random
import sys
from datetime import datetime
from pprint import pprint
from subprocess import call

from lp_ls.scalability_internal import scalability

test = False


def scaling_nof_facilities(command, \
                           objective, \
                           input_range, \
                           results_dir):
    print("scaling_nof_facilities", command, input_range)
    init_seed = 123456789
    random.seed(init_seed)

    nof_facilities = input_range
    nof_groups = [7]
    nof_centers = [5]
    nof_iterations = 2 if test else 5

    logfile_name = "./%s/scaling_nof_facilities-%s-%s.log" % \
                   (results_dir, command, objective)
    logfile = open(logfile_name, 'w')

    result_file_name = "./%s/scaling_nof_facilities-%s-%s.results" % \
                       (results_dir, command, objective)
    result_file = open(result_file_name, 'w')

    stats = scalability(nof_facilities, \
                        nof_centers, \
                        nof_iterations, \
                        nof_groups, \
                        objective,
                        command, \
                        logfile, \
                        result_file)
    pprint(stats, stream=logfile)


# end scaling_nof_facilities()


def scaling_nof_centers(command, \
                        objective, \
                        input_range, \
                        results_dir):
    print("scaling_nof_centers", command, input_range)
    init_seed = 123456789
    random.seed(init_seed)

    nof_facilities = [1000]
    nof_groups = [5]
    nof_centers = input_range
    nof_iterations = 2 if test else 5

    logfile_name = "./%s/scaling_nof_centers-%s-%s.log" % \
                   (results_dir, command, objective)
    logfile = open(logfile_name, 'w')
    if test:
        logfile = sys.stdout

    result_file_name = "./%s/scaling_nof_centers-%s-%s.results" % \
                       (results_dir, command, objective)
    result_file = open(result_file_name, 'w')

    stats = scalability(nof_facilities, \
                        nof_centers, \
                        nof_iterations, \
                        nof_groups, \
                        objective,
                        command, \
                        logfile, \
                        result_file)
    pprint(stats, stream=logfile)


# end scaling_nof_centers()


def scaling_nof_groups(command, \
                       objective, \
                       input_range, \
                       results_dir='exp-results'):
    print("scaling_nof_groups", command, input_range)
    init_seed = 123456789
    random.seed(init_seed)

    nof_facilities = [1000]
    nof_groups = input_range
    nof_centers = [5]
    nof_iterations = 2 if test else 5

    logfile_name = "./%s/scaling_nof_groups-%s-%s.log" % \
                   (results_dir, command, objective)
    logfile = open(logfile_name, 'w')

    result_file_name = "./%s/scaling_nof_groups-%s-%s.results" % \
                       (results_dir, command, objective)
    result_file = open(result_file_name, 'w')

    stats = scalability(nof_facilities, \
                        nof_centers, \
                        nof_iterations, \
                        nof_groups, \
                        objective,
                        command, \
                        logfile, \
                        result_file)
    pprint(stats, stream=logfile)


# end scaling_nof_groups()

def scaling_batch():
    now = datetime.now()
    results_dir_path = 'scaling-LP-LS/%s-%s-%s_%s-%s-%s' % \
                       (now.year, now.month, now.day, now.hour, now.minute, now.second)
    cmd = 'mkdir -p %s' % (results_dir_path)
    call(cmd, shell=True)

    print(results_dir_path)
    objective = 'kmedian'
    command = 'LP-LS'

    scaling_nof_facilities(command, objective, \
                           [100, 1000] if test else [100, 1000, 10000], \
                           results_dir_path)
    scaling_nof_centers(command, objective, \
                        [3, 4] if test else [4, 5, 6, 7, 8, 9, 10], \
                        results_dir_path)
    scaling_nof_groups(command, objective, \
                       [3, 4] if test else [4, 5, 6, 7], \
                       results_dir_path)


# end scaling_batch()


if __name__ == '__main__':
    scaling_batch()
