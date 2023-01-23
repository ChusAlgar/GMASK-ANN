from sklearn.neighbors import NearestNeighbors
import numpy as np
import logging


# Using Brute Force Algorithm, build the index of nearest neighbors
def bruteforce_nn_index(train_set, k, distance_type, algorithm):

    # Determine the knn of each element on the trainset
    knn_index = NearestNeighbors(n_neighbors=k, metric=distance_type, algorithm=algorithm).fit(train_set)
    logging.info("Building index")

    return knn_index


# Using Brute Force Algorithm and index built, find the k nearest neighbors
def bruteforce_nn_search(train_set, test_set, k, distance_type, algorithm, knn_index):

    # Found the knn of the testset between those contained on the trainset
    dists, indices = knn_index.kneighbors(test_set)

    coords = np.array(train_set[indices])

    # Return knn and their distances with the query points
    #logging.info(str(k) + "-Nearest Neighbors found using BruteForce + " + distance_type + " distance + " + algorithm + " algorithm.")

    return indices, coords, dists

