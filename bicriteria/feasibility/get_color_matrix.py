import os
import sys
import time

import numpy as np


def get_color_matrix(color_matrix_file, logfile):
    tstart = time.time()
    if not os.path.exists(color_matrix_file):
        sys.stderr.write("File '%s' do not exist\n" % (color_matrix_file))
    # end if
    color_matrix = np.loadtxt(fname=color_matrix_file, delimiter=",", dtype=np.uint16)

    logfile.write("get-color-matrix: [num-colors: %d] [total-time: %.2fs]\n" % \
                  (color_matrix.shape[0], time.time() - tstart))
    logfile.flush()

    return color_matrix
