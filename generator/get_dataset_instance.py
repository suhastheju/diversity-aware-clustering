from sklearn.datasets import make_blobs


def get_dataset_instance(N, d, k, seed=0, c_std=0.8):
    data, _ = make_blobs(n_samples=N,
                         centers=k,
                         n_features=d,
                         random_state=seed,
                         cluster_std=c_std)
    return data
