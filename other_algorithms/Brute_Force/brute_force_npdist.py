from sklearn.neighbors import NearestNeighbors
from timeit import default_timer as timer
import numpy as np


def brute_force_nn(train_set, test_set, k, distance_type, algorithm):

    start_time = timer()

    # Determine the knn of each element on the trainset
    knn = NearestNeighbors(n_neighbors=k, metric=distance_type, algorithm=algorithm).fit(train_set)

    # Found the knn of the testset between those contained on the trainset
    dists, indices = knn.kneighbors(test_set)

    end_time = timer()
    execution_time = end_time - start_time

    coords = np.array(train_set[indices])

    # Return knn and their distances with the query points
    print ("\n" + str(k) + "-Nearest Neighbors found using Brute Force + " + distance_type + " distance + " + algorithm + " algorithm.")

    return indices, coords, dists, execution_time

