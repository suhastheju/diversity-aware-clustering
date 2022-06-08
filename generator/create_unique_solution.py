import numpy as np


def create_unique_solution(rvec, groups, k):
    nof_facilities = len(groups[0])
    nof_groups = len(groups)

    # require 'k' facilities from group 't'
    rvec[len(rvec) - 1] = k

    for i in range(0, nof_groups):
        for j in range(nof_facilities - k, nof_facilities):
            groups[i][j] = 1
        # end for
        for j in range(0, nof_facilities - k):
            groups[0][j] = 0
        # end for
    # end for

    solution = []
    for j in range(nof_facilities - k, nof_facilities):
        solution.append(j + 1)
    # end for

    return rvec, groups, np.array(solution)
