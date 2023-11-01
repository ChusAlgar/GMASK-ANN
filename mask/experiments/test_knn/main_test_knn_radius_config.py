import logging
# import mask.kmeans_tree_npdist_modulado as ktd
import mask.KNN_np as knn
# import mask.data_test as dt
import numpy as np
from timeit import default_timer as timer
import experiments.load_train_test_set as lts
import experiments.neighbors_utils as sgn


def MASK(config):

    # Read test parameters
    dataset = config.get('test', 'dataset')
    k_vecinos = config.getint('test', 'k')
    metrica = config.get('test', 'distance')  # Possible values: euclidean
    method = config.get('test', 'method')
    tam_grupo = config.getint('method', 'tg')
    n_centroides = config.getint('method', 'nc')
    radio = config.get('method', 'r')
    algorithm = config.get('method', 'algorithm') # Possible values kmeans, kmedoids
    implementation = config.get('method', 'implementation')  # Possible values: pyclustering (elkan), sklearn (elkan), hammerly, kclust (?)


    # Set log configuration
    logging.basicConfig(filename="./experiments/logs/" + dataset + "/test_knn_" + dataset + "_" + str(k_vecinos) + "_" + metrica + "_" + method + "_tg" + str(tam_grupo) + "_nc" + str(n_centroides) + "_r" + str(radio) + "_" + str(algorithm) + "_" + str(implementation) + ".log", filemode='w', format='%(asctime)s - %(name)s - %(message)s', level=logging.INFO)
    logging.info('------------------------------------------------------------------------')
    logging.info('                          KNN Searching')
    logging.info('------------------------------------------------------------------------\n')
    logging.info("")
    logging.info("---- Searching the " + str(k_vecinos) + " nearest neighbors within " + method + " over " + str(
        dataset) + " dataset using " + str(metrica) + " distance. ----")
    logging.info("")
    logging.info('---- MASK Parameters - tam_grupo=%s - n_centroids=%s - radius=%s - algorithm=%s - implementation=%s ----', tam_grupo, n_centroides, radio, algorithm, implementation)

    # Regarding the dataset name, set the file name to load the train and test set
    file_name = "./data/" + str(dataset) + "_train_test_set.hdf5"


    # 1º Leemos los datos

    # Read train and test set from preprocesed h5py file
    vector_training, vector_testing = lts.load_train_test_h5py(file_name)

    # Read train and test set from original file
    # vector_training, vector_testing = lts.load_train_test(str(dataset))

    dimensionalidad = vector_testing.shape[1]
    cant_ptos = len(vector_training)


    # 2º Generamos el árbol
    n_capas, grupos_capa, puntos_capa, labels_capa = knn.mask_tree(cant_ptos, tam_grupo, n_centroides,
                                                                     metrica, vector_training, dimensionalidad, algorithm, implementation)

    # Store index in a file
    # with open("./other_algorithms/MASK_MNIST" + str(k_vecinos) +".pickle", 'wb') as handle:
    #    dump((puntos_capa, labels_capa), handle)


    # 3º Buscamos los k vecinos de los puntos de testing
    start_time_s = timer()

    # Creaamos las estructuras para almacenar los futuros vecinos
    indices_vecinos = np.empty([len(vector_testing), k_vecinos], dtype=int)
    coords_vecinos = np.empty([len(vector_testing), k_vecinos, vector_testing.shape[1]], dtype=float)
    dists_vecinos = np.empty([len(vector_testing), k_vecinos], dtype=float)

    for i in range(len(vector_testing)):

        punto = vector_testing[i]
        #logging.info('punto %s =%s', i,  punto)

        start_time_iter = timer()
        vecinos_i = knn.mask_radius_search(n_centroides, punto, vector_training, k_vecinos, metrica,
                                       grupos_capa, puntos_capa, labels_capa, dimensionalidad, float(radio))

        end_time_iter = timer()
        #logging.info('Index time= %s seconds', end_time_iter - start_time_iter)
        #logging.info('punto %s - time= %s seconds', i, end_time_iter - start_time_iter)

        indices_vecinos[i] = vecinos_i[0]
        coords_vecinos[i] = vecinos_i[1]
        dists_vecinos[i] = vecinos_i[2]


    end_time_s = timer()
    logging.info('Search time = %s seconds\n', end_time_s - start_time_s)
    logging.info('Average time spent in searching a single point = %s', (end_time_s - start_time_s)/vector_testing.shape[0])
    logging.info('Speed (points/s) = %s\n', vector_testing.shape[0]/(end_time_s - start_time_s))

    # Regarding the knn, method, dataset_name and distance choosen, set the file name to store the neighbors
    file_name = "./experiments/NearestNeighbors/" + dataset + "/knn_" + dataset + "_" + str(k_vecinos) + "_" + metrica + "_" + method + "_tg" + str(tam_grupo) + "_nc" + str(n_centroides) + "_r" + str(radio) + "_" + str(algorithm) + "_" + str(implementation) + ".hdf5"

    # Store indices, coords and dist into a hdf5 file
    sgn.save_neighbors(indices_vecinos, coords_vecinos, dists_vecinos, file_name)

    logging.info('------------------------------------------------------------------------\n')
