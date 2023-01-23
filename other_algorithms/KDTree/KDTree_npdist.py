from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KDTree
import numpy as np
import logging

# Using sklearn.neighbors.KDTree, build the index of nearest neighbors
def KDTree_nn_index(train_set, k, distance_type, algorithm):

    # Determine the knn of each element on the trainset
    tree_index = KDTree(train_set)
    logging.info("Building index")

    return tree_index


# Using sklearn.neighbors.KDTree to build the index, find the k nearest neighbors
def KDTree_nn_search(train_set, test_set, k, distance_type, algorithm, tree_index):

    # Found the knn of the testset between those contained on the trainset
    dists, indices = tree_index.query(test_set, k)

    coords = np.array(train_set[indices])

    # Return knn and their distances with the query points
    #logging.info(str(k) + "-Nearest Neighbors found using Brute Force + " + distance_type + " distance + " + algorithm + " algorithm.")

    return indices, coords, dists


# Using sklearn.neighbors.NearestNeighbors with KDTree Algorithm, build the index of nearest neighbors
def KDTree_nn_index_nn(train_set, k, distance_type, algorithm):

    # Determine the knn of each element on the trainset
    knn_index = NearestNeighbors(n_neighbors=k, metric=distance_type, algorithm=algorithm).fit(train_set)
    logging.info("Building index")

    return knn_index


# Using sklearn.neighbors.NearestNeighbors to build the index, find the k nearest neighbors
def KDTree_nn_search_nn(train_set, test_set, k, distance_type, algorithm, knn_index):

    # Found the knn of the testset between those contained on the trainset
    dists, indices = knn_index.kneighbors(test_set)

    coords = np.array(train_set[indices])

    # Return knn and their distances with the query points
    #logging.info(str(k) + "-Nearest Neighbors found using Brute Force + " + distance_type + " distance + " + algorithm + " algorithm.")

    return indices, coords, dists

