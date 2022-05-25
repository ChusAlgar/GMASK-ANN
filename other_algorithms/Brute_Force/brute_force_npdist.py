from sklearn.neighbors import NearestNeighbors
import numpy as np


def brute_force_nn(train_set, test_set, knn, distance_type, algorithm):

    knn = NearestNeighbors(n_neighbors=knn, metric=distance_type, algorithm=algorithm).fit(train_set)

    dists, indices = knn.kneighbors(test_set)
    coords = np.array(train_set[indices])

    return indices, coords, dists

