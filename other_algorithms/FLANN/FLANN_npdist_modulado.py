import numpy as np
from pyflann import *
import logging
from pickle import dump, load


def FLANN_nn_index(dataset, ncentroids, distance_type, algorithm):

    # Sets the distance type used. Possible values: euclidean, manhattan, minkowski, max_dist, hik, hellinger, cs, kl.
    set_distance_type(distance_type, order=0)

    # Create a FLANN instance and build and index
    flann = FLANN()
    flann.build_index(dataset, target_precision=0.9, log_level="info", algorithm=algorithm)

    # Using kmeans, compute the n-centroids describing the data
    centroids = flann.kmeans(dataset, num_clusters=ncentroids, max_iterations=None, mdtype=None)
    # print centroids

    # Store index built on disk to use it later on a file called 'index_'
    flann.save_index('./other_algorithms/FLANN/index_')
    logging.info("Saving FLANN index at 'index_'")

    # Store index on disk to obtain its size
    #with open("./other_algorithms/FLANN/MNIST_knn.pickle", 'wb') as handle:
        #dump(flann.nn_index, handle)
    

    return centroids


def FLANN_nn_search(dataset, seq_buscada, k, distance_type, algorithm):

    # Sets the distance type used. Possible values: euclidean, manhattan, minkowski, max_dist, hik, hellinger, cs, kl.
    set_distance_type(distance_type, order=0)

    # If there is an index stored on disk:
    if os.path.isfile('./other_algorithms/FLANN/index_'):

        # Create a FLANN instance
        flann = FLANN()

        # Load the index stored
        flann.load_index('./other_algorithms/FLANN/index_', dataset)
        #logging.info("Loading FLANN index from 'index_'...")

    else:
        logging.info("No FLANN index found. Creating a new one...")

        # Create a FLANN instance
        flann = FLANN_nn_index(dataset, 128, distance_type, algorithm)



    # Find the knn of each point in seq_buscada using this index
    lista_indices, lista_coords, lista_dists = [], [], []

    # For every point contained on the train set (the complete dataset in this case), find its k
    # nearest neighbors on this dataset using FLANN and the index built previously
    for f in range(seq_buscada.shape[0]):
        # print("Point number " + str(f))
        indices, dists = flann.nn_index(seq_buscada[f], num_neighbors=k, algorithm=algorithm)
        coords = np.array(dataset[indices])

        lista_indices.append(indices)
        lista_coords.append(coords)
        lista_dists.append(dists)


    # Return knn and their distances with the query points
    #logging.info(str(k) + "-Nearest Neighbors found using FLANN + " + distance_type + " distance + " + algorithm + " algorithm.")

    return np.array(lista_indices), np.array(lista_coords), np.array(lista_dists)
