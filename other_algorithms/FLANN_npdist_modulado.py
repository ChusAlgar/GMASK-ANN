import numpy
from pyflann import *
import data_test as dt


def FLANN_tree(dataset, ncentroids, normaliza, distance_type):

    # Sets the distance type used. Possible values: euclidean, manhattan, minkowski, max_dist, hik, hellinger, cs, kl.
    set_distance_type(distance_type, order=0)

    # Create a FLANN instance and build and index
    flann = FLANN()
    flann.build_index(dataset, target_precision=0.9, log_level="info")

    # Store index built on disk to use it later on a file called 'index_'
    flann.save_index('index_')
    print("Saving FLANN index at 'index_'")

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

    else:
        print("No FLANN index found. Creating a new one...")
        # Find the knn of each point in seq_buscada creating a new index
        result, dists = flann.nn(dataset, seq_buscada, num_neighbors=knn, algorithm=algorithm) # branching=32, iterations=7, checks=1

    # Return knn and their distances with the query points
    print result
    print dists

    # Count number of hit and miss (hit = the knn returned is exactly the point searched)
    hit = 0.0
    miss = 0.0

    for query_pt in dists:
        for neighbor in query_pt:
            if neighbor == 0:
                hit = hit+1.0
            else:
                miss = miss+1.0

    # Show percentage of ok/ko on screen
    print("Porcentaje de aciertos: ", hit/dists.size * 100)
    print("Porcentaje de fallos: ", miss/dists.size * 100)

    return result


