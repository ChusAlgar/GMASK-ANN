import numpy as np
from pyflann import *
import data_test as dt



def FLANN_tree(dataset, ncentroids, distance_type, algorithm):

    # Sets the distance type used. Possible values: euclidean, manhattan, minkowski, max_dist, hik, hellinger, cs, kl.
    set_distance_type(distance_type, order=0)

    # Create a FLANN instance and build and index
    flann = FLANN()
    flann.build_index(dataset, target_precision=0.9, log_level="info", algorithm=algorithm)

    # Store index built on disk to use it later on a file called 'index_'
    flann.save_index('index_')
    print("\nSaving FLANN index at 'index_'")

    # Using kmeans, compute the ncentroids describing the data
    centroids = flann.kmeans(dataset, num_clusters=ncentroids, max_iterations=None, mdtype=None)
    #print centroids

    return centroids


def FLANN_nn(dataset, seq_buscada, knn, distance_type, algorithm):

    # Sets the distance type used. Possible values: euclidean, manhattan, minkowski, max_dist, hik, hellinger, cs, kl.
    set_distance_type(distance_type, order=0)

    # Create a FLANN instance
    flann = FLANN()

    # If there is an index stored on disk:
    if os.path.isfile('index_'):
        # Load it
        flann.load_index('index_', dataset)
        print("Loading FLANN index from 'index_'...")
        # Find the knn of each point in seq_buscada using this index
        result, dists = flann.nn_index(seq_buscada, num_neighbors=knn, algorithm=algorithm)
        coords = np.array(dataset[result])

    else:
        print("No FLANN index found. Creating a new one...")
        # Find the knn of each point in seq_buscada creating a new index
        result, dists = flann.nn(dataset, seq_buscada, num_neighbors=knn, algorithm=algorithm) # branching=32, iterations=7, checks=1
        coords = np.array(dataset[result])

    # Return knn and their distances with the query points
    print ("\nK-Nearest Neighbors found using FLANN + " + distance_type + " distance + " + algorithm + " algorithm.")
    #print result

    return result, coords, dists

# Deprecated
'''
def benchmark(dataset, seq_buscada, nn, dists, distance_type):

    # Sets the distance type used. Possible values: euclidean, manhattan, minkowski, max_dist, hik, hellinger, cs, kl.
    set_distance_type(distance_type, order=0)

    # Create a FLANN instance
    flann = FLANN()

    # Find the "optimus" knn of each point in seq_buscada creating a new index through linear exploration
    nn_op, dists_op = flann.nn(dataset, seq_buscada, num_neighbors=len(nn[0]), algorithm='linear')  # branching=32, iterations=7, checks=1

    print ("\nK-Nearest Neighbors found using linear exploration (optimus solution):")
    print nn_op

    # Count number of hit and miss (hit = the nn returned is part of the set of "optimums")
    hit_nn = 0.0
    miss_nn = 0.0

    for i in range(len(seq_buscada)):
        for j in range(len(nn[0])):

            if nn[i][j] in nn_op[i]:
                hit_nn = hit_nn+1.0
            else:
                miss_nn = miss_nn+1.0

    # Recall: %  hit returned vs number of points
    # Show percentage of hit/miss on screen
    print ("\nBenchmark:")
    print("Porcentaje de aciertos: " + str(hit_nn/nn.size * 100))
    print("Porcentaje de fallos: " + str(miss_nn/nn.size * 100))

    return hit_nn, miss_nn
'''

