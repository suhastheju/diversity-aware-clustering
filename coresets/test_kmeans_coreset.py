from sklearn.datasets import make_blobs

from coresets.kmeans_coreset import kmeans_coreset


def test_kmeans_coreset():
    N = 100
    d = 4
    K = 4

    data, _ = make_blobs(n_samples=N,
                         centers=K,
                         n_features=d,
                         random_state=0,
                         cluster_std=0.8)

    coreset_size = 0.1
    stats = kmeans_coreset(data, K, coreset_size)
    print(stats)
