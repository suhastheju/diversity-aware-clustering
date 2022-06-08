import math
import random
import sys

import numpy as np


def generate_groups(nof_groups=5, \
                    nof_facilities=10, \
                    max_freq=3, \
                    seed=123456789):
    # error handling
    if (max_freq > nof_groups):
        sys.stderr.write("invalied parameter values: mu = %d, t = %d\n" % \
                         (max_freq, nof_groups))
    # end if

    random.seed(seed)
    result = np.zeros((nof_groups, nof_facilities), dtype=np.int8)
    for j in range(0, nof_facilities):
        freq = min(math.ceil(random.random() * max_freq), nof_groups)
        groups = random.sample(range(nof_groups), freq)
        for i in groups:
            result[i, j] = 1
        # end for
    # end for
    return result
