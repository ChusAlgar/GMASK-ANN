from other_algorithms.load_train_test_set import *
from other_algorithms.neighbors_utils import *
from other_algorithms.BruteForce.bruteforce_npdist import bruteforce_nn_index, bruteforce_nn_search
import logging
from timeit import default_timer as timer

# Set constants for experiments
algorithm = 'brute'  # Possible values 'auto', 'ball_tree', 'kd_tree', 'brute'
ncentroids = 128  # At MASK, ncentroids = tam_grupo*n_centroides = 8*16 = 128
ks = [5, 10, 15]  # Number of Nearest Neighbors to be found
datasets = ['gaussian', 'municipios', 'MNIST']
distances = ['euclidean', 'manhattan', 'chebyshev']   # Possible values:
                             # From scikit-learn: ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan']. These metrics support sparse matrix inputs. ['nan_euclidean'] but it does not yet support sparse matrices
                             # From scipy.spatial.distance: ['braycurtis', 'canberra', 'chebyshev', 'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule']bThese metrics do not support sparse matrix inputs.

# Set log configuration
logging.basicConfig(filename='./logs/results_knn_BruteForce.log',
                    filemode='w', format='%(asctime)s - %(name)s - %(message)s', level=logging.INFO)
logging.info('------------------------------------------------------------------------')
logging.info('KNN Search using Brute Force Algorithm')
logging.info('------------------------------------------------------------------------\n')


# Load the train and test sets of each dataset to carry on the experiments
for dataset_name in datasets:

    logging.info('-------------------  %s Dataset  -------------------------------\n', dataset_name)

    # Regarding the dataset_name set the file name to load the train and test set
    file_name = "./data/" + str(dataset_name) + "_train_test_set.hdf5"

    # train_set, test_set = load_train_test(file_name)
    train_set, test_set = load_train_test_h5py(file_name)


    # Find the knn from the train_set of the elements contained in the test_set, using distances choosen
    for d in distances:

        logging.info('------------  %s distance  --------------------\n', d)

        for k in ks:

            # Using Brute Force Algorithm, build the index of nearest neighbors
            start_time_i = timer()
            knn_index = bruteforce_nn_index(train_set, k, d, algorithm)
            end_time_i = timer()
            logging.info('Index time= %s seconds', end_time_i - start_time_i)

            # Using Brute Force Algorithm and the index built, find the k nearest neighbors
            start_time_s = timer()
            indices, coords, dists = bruteforce_nn_search(train_set, test_set, k, d, algorithm, knn_index)
            end_time_s = timer()
            logging.info('Search time= %s seconds\n', end_time_s - start_time_s)

            # Regarding the knn, method, dataset_name and distance choosen, set the file name to store the neighbors
            file_name = "./NearestNeighbors/BruteForce/" + str(dataset_name) + "_" + str(
                d) + "_BruteForce_" + str(k) + "nn.hdf5"

            # Store indices, coords and dist into a hdf5 file
            save_neighbors(indices, coords, dists, file_name)
            # logging.info('Neighbors stored')

            # Print
            # print_knn(train_set, test_set, coords, dataset_name, d, "BruteForce", k)

logging.info('------------------------------------------------------------------------\n')
