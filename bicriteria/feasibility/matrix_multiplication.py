import time


def matrix_multiplication(A, B, logfile):
    tstart = time.time()
    C = A.dot(B)
    logfile.write("matrix-multiplication: [input-size: (%d, %d) X (%d, %d)]" % \
                  (A.shape[0], A.shape[1], B.shape[0], B.shape[1]))
    logfile.write(" [output-size: (%d, %d)]" % (C.shape[0], C.shape[1]))
    logfile.write(" [total-time: %.2fs]\n" % (time.time() - tstart))
    logfile.flush()
    return C
