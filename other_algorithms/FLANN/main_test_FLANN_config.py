from other_algorithms.load_train_test_set import *
from other_algorithms.neighbors_utils import *
from other_algorithms.FLANN.FLANN_npdist_modulado import FLANN_nn_index, FLANN_nn_search
import logging
from timeit import default_timer as timer
import ConfigParser
import io

def FLANN(file):

    # Load the configuration file
    configfile_name = "./config/" + file

    with open(configfile_name) as f:
        config_file = f.read()
    config = ConfigParser.RawConfigParser(allow_no_value=True)
    config.readfp(io.BytesIO(config_file))


    # Read test parameters
    dataset = config.get('test', 'dataset')
    k = config.getint('test', 'k')
    distance = config.get('test', 'distance')  # Possible values euclidean, manhattan, minkowski, max dist (L infinity), hik (histogram intersection kernel), hellinger, cs (chi-square) and kl (Kullback- Leibler)
    method = config.get('test', 'method')
    ncentroids = config.getint('method', 'ncentroids')  # At MASK, ncentroids = tam_grupo*n_centroides = 8*16 = 128
    algorithm = config.get('method', 'algorithm')  # Possible values: linear, kdtree, kmeans, composite, autotuned - default: kdtree


    # Set log configuration
    logging.basicConfig(filename="./logs/test_knn_"  + dataset + "_" + str(k) + "_" + distance + "_" + method + ".log",
                        filemode='w', format='%(asctime)s - %(name)s - %(message)s', level=logging.INFO)
    logging.info('------------------------------------------------------------------------')
    logging.info('                          KNN Searching')
    logging.info('------------------------------------------------------------------------\n')
    logging.info("")
    logging.info("---- Searching the " + str(k) + " nearest neighbors within " + method + " over " + str(
        dataset) + " dataset using " + str(distance) + " distance. ----")



    # Regarding the dataset name, set the file name to load the train and test set
    file_name = "./data/" + str(dataset) + "_train_test_set.hdf5"

    # Load the train and test sets of each dataset to carry on the experiments
    # train_set, test_set = load_train_test(str(dataset))
    train_set, test_set = load_train_test_h5py(file_name)


    # GENERATE INDEX AND CENTROIDS
    # AND FIND THE KNN FROM THE train_set OF THE ELEMENTS CONTAINED IN THE test_set, USING DISTANCE CHOOSEN


    # Using FLANN, build the index tree and generate the num_centroids describing the data
    start_time_i = timer()
    centroids = FLANN_nn_index(train_set, ncentroids, distance, algorithm)
    end_time_i = timer()
    logging.info('Index time= %s seconds',  end_time_i - start_time_i)

    # Using FLANN and the index built, search for the knn nearest neighbors
    start_time_s = timer()
    indices, coords, dists = FLANN_nn_search(train_set, test_set, k, distance, algorithm)
    end_time_s = timer()
    logging.info('Search time = %s seconds\n', end_time_s - start_time_s)
    logging.info('Average time spent in searching a single point = %s', (end_time_s - start_time_s)/test_set.shape[0])
    logging.info('Speed (points/s) = %s\n', test_set.shape[0]/(end_time_s - start_time_s))

    # Store indices, coords and dist into a tridimensional matrix of size vector.size() x 3 x knn
    # knn = zip(indices, coords, dists)

    # Regarding the knn, method, dataset_name and distance choosen, set the file name to store the neighbors
    file_name = "./NearestNeighbors/knn_" + dataset + "_" + str(k) + "_" + distance + "_" + method + ".hdf5"
    # Store indices, coords and dist into a hdf5 file
    save_neighbors(indices, coords, dists, file_name)

    # Print
    # print_knn(train_set, test_set, coords, dataset_name, d, "FLANN", k)

    '''
    # Obtain error rate of the K Nearest Neighbors found
    file_name_le = "./NearestNeighbors/knn_" + dataset + "_" + str(k) + "_" + distance + "_Brute_Force.hdf5"
    file_name = "./NearestNeighbors/knn_" + dataset + "_" + str(k) + "_" + distance + "_" + method + ".hdf5"
    
    error_rate(dataset, distance, 'FLANN', k, False, file_name_le, file_name)
    '''

    logging.info('------------------------------------------------------------------------\n')
