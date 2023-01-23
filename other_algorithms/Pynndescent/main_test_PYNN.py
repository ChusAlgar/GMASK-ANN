from other_algorithms.load_train_test_set import *
from other_algorithms.neighbors_utils import *
from other_algorithms.Pynndescent.PYNN_npdist import PYNN_nn_index, PYNN_nn_search
import logging
from timeit import default_timer as timer
from pickle import dump, load



# Set constants for experiments
ks = [5, 10, 15]  # Number of Nearest Neighbors to be found
datasets = ['MNIST']
distances = ['euclidean']   # Possible values euclidean, manhattan, minkowski, max dist (L infinity), hik (histogram intersection kernel), hellinger, cs (chi-square) and kl (Kullback- Leibler)


# Load the train and test sets of each dataset to carry on the experiments
for dataset_name in datasets:

    # Set log configuration
    logging.basicConfig(filename="./logs/knn_" + dataset_name +"_PYNN.log",
                        filemode='w', format='%(asctime)s - %(name)s - %(message)s', level=logging.INFO)
    logging.info('------------------------------------------------------------------------')
    logging.info('KNN Search using PYNN')
    logging.info('------------------------------------------------------------------------\n')

    logging.info('-------------------  %s Dataset  -------------------------------\n', dataset_name)

    # Regarding the dataset_name set the file name to load the train and test set
    file_name = "./data/" + str(dataset_name) + "_train_test_set.hdf5"

    # train_set, test_set = load_train_test(file_name)
    train_set, test_set = load_train_test_h5py(file_name)


    # Generate index and centroids
    # and find the knn from the train_set of the elements contained in the test_set, using distances choosen
    for d in distances:

        logging.info('------------  %s distance  --------------------\n', d)

        for k in ks:

            logging.info("")
            logging.info("---- Case " + str(k) + " nn within PYNN over " + str(
                dataset_name) + " dataset using " + str(d) + " distance. ----")

            # Using PYNN, build the index tree and generate the num_centroids describing the data
            start_time_i = timer()
            pynn_index = PYNN_nn_index(train_set, d)
            end_time_i = timer()
            logging.info('Index time= %s seconds',  end_time_i - start_time_i)

            # Store index on disk to obtain its size
            #with open("./other_algorithms/Pynndescent/" + dataset_name + str(k) +".pickle", 'wb') as handle:
            #    dump(pynn_index, handle)


            # Using PYNN and the index built, search for the knn nearest neighbors
            start_time_s = timer()
            indices, coords, dists = PYNN_nn_search(train_set, test_set, k, d, pynn_index)
            end_time_s = timer()
            logging.info('Search time = %s seconds\n', end_time_s - start_time_s)
            logging.info('Average time spended in searching a single point = %s', (end_time_s - start_time_s)/test_set.shape[0])
            logging.info('Speed (points/s) = %s\n', test_set.shape[0]/(end_time_s - start_time_s))

            # Store indices, coords and dist into a tridimensional matrix of size vector.size() x 3 x knn
            # knn = zip(indices, coords, dists)

            # Regarding the knn, method, dataset_name and distance choosen, set the file name to store the neighbors
            file_name = "./NearestNeighbors/PYNN/" + str(dataset_name) + "_" + str(
                d) + "_PYNN_" + str(k) + "nn.hdf5"

            # Store indices, coords and dist into a hdf5 file
            save_neighbors(indices, coords, dists, file_name)

            # Print
            # print_knn(train_set, test_set, coords, dataset_name, d, "PYNN", k)

            file_name_le = "./NearestNeighbors/Brute_Force/" + str(dataset_name) + "_" + str(
                d) + "_Brute_Force_" + str(k) + "nn.hdf5"
            file_name = "./NearestNeighbors/PYNN/" + str(dataset_name) + "_" + str(
                d) + "_PYNN_" + str(k) + "nn.hdf5"


            error_rate(dataset_name, d, 'PYNN', k, False, file_name_le, file_name)

        

logging.info('------------------------------------------------------------------------\n')
