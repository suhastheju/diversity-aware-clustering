import argparse
import sys
import time

import numpy as np

from bicriteria.feasibility.feasibility import calculate
from bicriteria.feasibility.get_color_matrix import get_color_matrix


def cmd_parser():
    parser = argparse.ArgumentParser()

    g = parser.add_argument_group('Arguments')
    # g.add_argument('-distances', '--distance-matrix', nargs='?', required=False,\
    #               type=str, default='distances.csv')
    g.add_argument('-colors', '--color-matrix', nargs='?', required=False, \
                   type=str, default='colors.csv')
    g.add_argument('-k', '--num-of-medians', nargs='?', required=True, type=int, \
                   default=2)
    g.add_argument('-rvec', '--requirements', nargs='+', \
                   required=True, type=int, default=[1, 1])
    g.add_argument('-cmd', '--command', nargs='?', required=True, type=str, \
                   default='all', choices=['all', 'brute-force', 'dynamic-program', 'linear-program'])
    g.add_argument('-log', '--log-file', nargs='?', required=False, type=str, \
                   default='stdout')
    return parser


def main():
    tstart = time.time()

    ## handling command line arguments
    parser = cmd_parser()
    opts = vars(parser.parse_args())
    # dist_matrix_file  = opts['distance_matrix']
    color_matrix_file = opts['color_matrix']
    num_of_medians = opts['num_of_medians']
    requirement_list = np.array(opts['requirements'], dtype='uint8')
    command = opts['command']
    output_filename = opts['log_file']

    if output_filename == 'stdout':
        logfile = sys.stdout
    else:
        logfile = open(output_filename, 'a')
    # end if
    logfile.write('\n\n')
    logfile.flush()

    # read color matrix
    color_matrix = get_color_matrix(color_matrix_file, logfile)

    # find a feasible solution
    calculate(num_of_medians, requirement_list, color_matrix, command)


## program entry point
if __name__ == "__main__":
    main()
