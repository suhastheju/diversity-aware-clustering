import random
import sys

import numpy as np

from generator.create_unique_solution import create_unique_solution
from generator.generate_groups import generate_groups
from generator.generate_requirements import generate_requirements


def get_feasible_instance(nof_groups, \
                          nof_facilities, \
                          r_max, \
                          r_min, \
                          max_freq, \
                          k, \
                          seed, \
                          unique=False):
    # error handling
    if (r_max > k or r_min < 1):
        sys.stderr.write("invalid parameter value: r_max = %d, r_min = %d, k = %d\n" % \
                         (r_max, r_min, k))
    # end if
    if (max_freq > nof_groups):
        sys.stderr.write("invalid parameter value: mu = %d, t = %d\n" % \
                         (max_freq, nof_groups))
    # end if

    # initiliaze random number generator
    random.seed(seed)
    rvec_seed = random.randint(1, sys.maxsize)
    groups_seed = random.randint(1, sys.maxsize)

    rvec = generate_requirements(nof_groups, r_max, r_min, rvec_seed)
    groups = generate_groups(nof_groups, nof_facilities, max_freq, groups_seed)

    solution = np.zeros(k)
    if (unique):
        rvec, groups, solution = create_unique_solution(rvec, groups, k)
    # end if

    return groups, rvec, solution
