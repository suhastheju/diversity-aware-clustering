# python imports
import argparse
import sys
from datetime import datetime
from subprocess import call

# local imports
import generator.get_distance_matrix
import generator.get_feasible_instance
from scalability_feasibility import test_batch_scaling
from scalability_optimal import test_batch_scaling_optimal


def test():
    N = 10
    seed = 1234

    dist_matrix = generator.get_distance_matrix.get_distance_matrix(N, seed)
    print(dist_matrix)


# end test()

###############################################################################
def cmd_parser():
    parser = argparse.ArgumentParser()
    g = parser.add_argument_group('Arguments')
    g.add_argument('-batch', '--batch-type', nargs='?', required=False, type=str, \
                   default='feasibility', choices=['feasibility', 'optimal', 'bicriteria'])
    g.add_argument('-obj', '--objective', nargs='?', required=False, type=str, \
                   default=None, choices=['kmedian', 'kmeans'])
    g.add_argument('-results', '--results-dir', nargs='?', required=False, type=str, \
                   default='exp-results')
    g.add_argument('-test', '--test-run', action='store_true')
    return parser


# end cmd_parser()


def main():
    parser = cmd_parser()
    opts = vars(parser.parse_args())
    batch_type = opts['batch_type']
    objective = opts['objective']
    results_dir = opts['results_dir']
    test_run = opts['test_run']

    if batch_type == 'bicriteria' and objective == None:
        sys.stderr.write("Error: specify objective function\n")
        sys.exit()
    # end if

    # create directory to store experimental results
    now = datetime.now()
    results_dir_path = '%s/%s-%s-%s_%s-%s-%s' % \
                       (results_dir, now.year, now.month, now.day, now.hour, now.minute, now.second)
    cmd = 'mkdir -p %s' % (results_dir_path)
    call(cmd, shell=True)

    # schedule experiments
    if batch_type == 'feasibility':
        test_batch_scaling(objective, results_dir_path, test_run)
    elif batch_type == 'optimal':
        test_batch_scaling_optimal(results_dir_path, test_run)
    elif batch_type == 'bicriteria':
        test_batch_scaling(objective, results_dir_path, test_run)
        test_batch_scaling_optimal(results_dir_path)


# end main()

## program entry point
if __name__ == "__main__":
    main()
# main()
