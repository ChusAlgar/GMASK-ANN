import numpy as np
from pyflann import *
from timeit import default_timer as timer
import data_test as dt



def FLANN_tree(dataset, ncentroids, distance_type, algorithm):

    # Sets the distance type used. Possible values: euclidean, manhattan, minkowski, max_dist, hik, hellinger, cs, kl.
    set_distance_type(distance_type, order=0)

    start_time = timer()

    # Create a FLANN instance and build and index
    flann = FLANN()
    flann.build_index(dataset, target_precision=0.9, log_level="info", algorithm=algorithm)

    # Using kmeans, compute the ncentroids describing the data
    centroids = flann.kmeans(dataset, num_clusters=ncentroids, max_iterations=None, mdtype=None)
    #print centroids

    end_time = timer()
    tree_time = end_time - start_time

    # Store index built on disk to use it later on a file called 'index_'
    flann.save_index('../other_algorithms/FLANN/index_')
    #print("\nSaving FLANN index at 'index_'")

    return centroids, tree_time


def FLANN_nn(dataset, seq_buscada, k, distance_type, algorithm):

    # Sets the distance type used. Possible values: euclidean, manhattan, minkowski, max_dist, hik, hellinger, cs, kl.
    set_distance_type(distance_type, order=0)

    # If there is an index stored on disk:
    if os.path.isfile('../other_algorithms/FLANN/index_'):

        start_time = timer()

        # Create a FLANN instance
        flann = FLANN()

        # Load the index stored
        flann.load_index('../other_algorithms/FLANN/index_', dataset)
        #print("Loading FLANN index from 'index_'...")

        # Find the knn of each point in seq_buscada using this index
        result, dists = flann.nn_index(seq_buscada, num_neighbors=k, algorithm=algorithm)
        coords = np.array(dataset[result])

    else:
        print("No FLANN index found. Creating a new one...")

        start_time = timer()

        # Create a FLANN instance
        flann = FLANN()

        # Find the knn of each point in seq_buscada creating a new index
        result, dists = flann.nn(dataset, seq_buscada, num_neighbors=k, algorithm=algorithm) # branching=32, iterations=7, checks=1
        coords = np.array(dataset[result])

    end_time = timer()
    search_time = end_time - start_time

    # Return knn and their distances with the query points
    print ("\n" + str(k) + "-Nearest Neighbors found using FLANN + " + distance_type + " distance + " + algorithm + " algorithm.")

    return result, coords, dists, search_time
