def parse_stats_LS(result):
    return result.astype({'n': 'int32', \
                          'k': 'int32', \
                          'swaps': 'int32', \
                          'dist_matrix_seed': 'int64', \
                          'local_search_seed': 'int64', \
                          'dist_matrix_time': 'float64', \
                          'objective_time': 'float64', \
                          'virtual_memory': 'float64', \
                          'peak_memory': 'float64'})
