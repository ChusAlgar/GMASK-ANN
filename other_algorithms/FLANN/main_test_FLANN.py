import sys
from load_train_test_set import *
from neighbors_utils import *
from FLANN_npdist_modulado import FLANN_tree, FLANN_nn


# Set constants for experiments
algorithm = 'kmeans'  # Possible values: linear, kdtree, kmeans, composite, autotuned - default: kdtree
ncentroids = 128  # At MASK, ncentroids = tam_grupo*n_centroides = 8*16 = 128
ks = [5, 10, 15]  # Number of Nearest Neighbors to be found
datasets = ["municipios"]
distances = ['euclidean', 'manhattan']   # Possible values euclidean, manhattan, minkowski, max dist (L infinity), hik (histogram intersection kernel), hellinger, cs (chi-square) and kl (Kullback- Leibler)


output_path = '../other_algorithms/FLANN/knn_FLANN_times.txt'

with open(output_path, 'w') as h:
    sys.stdout = h

    # Load the train and test sets of each dataset to carry on the experiments
    for dataset_name in datasets:

        # train_set, test_set = load_train_test(dataset_name)
        train_set, test_set = load_train_test_h5py(dataset_name)


        # Generate index and centroids
        #  and find the knn from the train_set of the elements contained in the test_set, using distances choosen
        for d in distances:

            for k in ks:

                # Using FLANN, build the index tree and generate the num_centroids describing the data
                centroids, tree_time = FLANN_tree(train_set, ncentroids, d, algorithm)

                # Using FLANN, find the knn nearest neighbors
                indices, coords, dists, search_time = FLANN_nn(train_set, test_set, k, d, algorithm)
                #print(indices)

                # Show time required to build the tree and search the knn
                print("Execution time = " + str(tree_time+search_time) + " seconds")

                # Store indices, coords and dist into a tridimensional matrix of size vector.size() x 3 x knn
                #knn = zip(indices, coords, dists)

                # Store indices, coords and dist into a hdf5 file
                save_neighbors(indices, coords, dists, dataset_name, d, "FLANN", k)

                # Print
                #print_knn(train_set, test_set, coords, dataset_name, d, "FLANN", k)
