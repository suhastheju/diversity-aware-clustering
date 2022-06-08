import time

import psutil
from sklearn_extra.cluster import KMedoids


def kmedoids_sklearn(data, k, iterations=300):
    # data: n X d numpy matrix
    # k: number of clusters
    # iterations: maximum number of iterations

    tstart = time.time()
    process = psutil.Process()

    km = KMedoids(n_clusters=k, max_iter=iterations).fit(data)
    medoid_indices = km.medoid_indices_
    cluster_centers = km.cluster_centers_
    inertia = km.inertia_

    total_time = time.time() - tstart

    stats = {}
    stats['cost'] = inertia
    stats['cluster_centers'] = medoid_indices
    stats['total_time'] = total_time
    stats['peak_memory'] = process.memory_info().rss / (1024 * 1024)
    stats['virtual_memory'] = process.memory_info().vms / (1024 * 1024)

    return stats
# end kmedoids_sklearn
