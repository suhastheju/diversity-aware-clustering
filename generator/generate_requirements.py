import math
import random
import sys

import numpy as np


def generate_requirements(nof_groups=5, \
                          r_max=3, \
                          r_min=1, \
                          seed=123456789):
    # error handling
    if (r_max > nof_groups or r_min < 1):
        sys.stderr.write("invalid parameter values: r_max = %d, r_min = %d, t = %d" % \
                         (r_max, r_min, nof_groups))
    # end if

    random.seed(seed)
    result = []
    for i in range(0, nof_groups):
        # square distribution
        a = r_min + math.floor((1 - math.sqrt(random.random())) * (r_max - r_min + 1))
        result.append(a)
    # end for
    return np.array(result)
