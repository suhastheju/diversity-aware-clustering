import time

import psutil


def debug_details(process, tstart, processed_matrix, vector_matrix, logfile):
    logfile.write("=======================================================================\n")
    logfile.write("RUN STATISTICS: FEASIBILITY\n")
    logfile.write("--------------\n")
    logfile.write("CPU:           [CORES-TOTAL: %s, THREADS-TOTAL: %s, THREADS-USED: %s]\n" % \
                  (psutil.cpu_count(logical=False), psutil.cpu_count(), process.num_threads()))
    logfile.write("\t       [CUR-FREQ: %f, MIN-FREQ: %f, MAX-FREQ: %f]\n" % \
                  (psutil.cpu_freq().current, psutil.cpu_freq().min, \
                   psutil.cpu_freq().max))
    logfile.write("MEMORY:        [RAM-MEM: %.2fMB, VIR-MEM: %.2fMB, TOTAL: %.2fMB]\n" % \
                  (process.memory_info().rss / (1024 * 1024), \
                   process.memory_info().vms / (1024 * 1024), \
                   (process.memory_info().rss + process.memory_info().vms) / (1024 * 1024)))
    logfile.write("MATRIX-MEMORY: [PROCESSED: %.2fMB, VECTOR: %.2fMB]\n" % \
                  (processed_matrix.nbytes / (1024 * 1024), \
                   vector_matrix.nbytes / (1024 * 1024)))
    logfile.write("TOTAL-TIME:    %.2fs\n" % (time.time() - tstart))
    logfile.write("======================================================================\n")
    logfile.flush()
