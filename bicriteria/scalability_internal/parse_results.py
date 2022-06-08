def parse_result(result):
    return result.astype({'n': 'int32', \
                          't': 'int32', \
                          'k': 'int32', \
                          'r_min': 'int32', \
                          'r_max': 'int32', \
                          'max_freq': 'int32', \
                          'seed': 'int64', \
                          'instance_time': 'float64', \
                          'feasibility_time': 'float64', \
                          'objective_time': 'float64'})
