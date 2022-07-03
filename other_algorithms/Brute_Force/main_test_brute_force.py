from load_train_test_set import *
from neighbors_utils import *
from brute_force_npdist import brute_force_nn
from scipy.spatial import distance
import contextlib
import sys

# Set constants for experiments
algorithm = 'brute'  # Possible values 'auto', 'ball_tree', 'kd_tree', 'brute'
ncentroids = 128  # At MASK, ncentroids = tam_grupo*n_centroides = 8*16 = 128
ks = [5, 10, 15]  # Number of Nearest Neighbors to be found
datasets = ['gaussian', 'municipios', 'MNIST']
distances = ['euclidean', 'manhattan', 'chebyshev']   # Possible values:
                             # From scikit-learn: ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan']. These metrics support sparse matrix inputs. ['nan_euclidean'] but it does not yet support sparse matrices
                             # From scipy.spatial.distance: ['braycurtis', 'canberra', 'chebyshev', 'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule']bThese metrics do not support sparse matrix inputs.

output_path = '../other_algorithms/Brute_Force/knn_brute_force_times.txt'

with open(output_path, 'w') as h:
    sys.stdout = h

    # Load the train and test sets of each dataset to carry on the experiments
    for dataset_name in datasets:

        # train_set, test_set = load_train_test(dataset_name)
        train_set, test_set = load_train_test_h5py(dataset_name)


        # Find the knn from the train_set of the elements contained in the test_set, using distances choosen
        for d in distances:

            for k in ks:

                # Using Brute Force Algorithm, find the k nearest neighbors
                indices, coords, dists, execution_time = brute_force_nn(train_set, test_set, k, d, algorithm)

                # Show time required to build the tree and search the knn
                print("Execution time = " + str(execution_time) + " seconds")

                # Store indices, coords and dist into a hdf5 file
                save_neighbors(indices, coords, dists, dataset_name, d, "Brute_Force", k)

                # Print
                #print_knn(train_set, test_set, coords, dataset_name, d, "Brute_Force", k)

