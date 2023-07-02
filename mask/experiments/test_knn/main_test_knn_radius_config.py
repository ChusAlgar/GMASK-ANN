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


    # Set log configuration
    logging.basicConfig(filename="./experiments/logs/" + dataset + "/test_knn_" + dataset + "_" + str(k_vecinos) + "_" + metrica + "_" + method + "_tg" + str(tam_grupo) + "_nc" + str(n_centroides) + "_r" + str(radio) + ".log", filemode='w', format='%(asctime)s - %(name)s - %(message)s', level=logging.INFO)
    logging.info('------------------------------------------------------------------------')
    logging.info('                          KNN Searching')
    logging.info('------------------------------------------------------------------------\n')
    logging.info("")
    logging.info("---- Searching the " + str(k_vecinos) + " nearest neighbors within " + method + " over " + str(
        dataset) + " dataset using " + str(metrica) + " distance. ----")
    logging.info("")
    logging.info('---- MASK Parameters - tam_grupo=%s - n_centroids=%s - radius=%s ----', tam_grupo, n_centroides, radio)

    # Regarding the dataset name, set the file name to load the train and test set
    file_name = "./data/" + str(dataset) + "_train_test_set.hdf5"


    # 1º Leemos los datos

    # Read train and test set from preprocesed h5py file
    vector_training, vector_testing = lts.load_train_test_h5py(file_name)

    # Read train and test set from original file
    # vector_training, vector_testing = lts.load_train_test(str(dataset))

    dimensiones = vector_testing.shape[1]
    cant_ptos = len(vector_training)


    # 2º Generamos el árbol
    n_capas, grupos_capa, puntos_capa, labels_capa = knn.kmeans_tree(cant_ptos, tam_grupo, n_centroides,
                                                                     metrica, vector_training, dimensiones)

    # Store index in a file
    # with open("./other_algorithms/MASK_MNIST" + str(k_vecinos) +".pickle", 'wb') as handle:
    #    dump((puntos_capa, labels_capa), handle)


    # 3º Buscamos los k vecinos de los puntos de testing
    start_time_s = timer()

    indices = np.empty([len(vector_testing), k_vecinos], dtype=int)
    coords = np.empty([len(vector_testing), k_vecinos, vector_testing.shape[1]], dtype=float)
    dists = np.empty([len(vector_testing), k_vecinos], dtype=float)

    for i in range(len(vector_testing)):
    # for i in range(1):
        punto = vector_testing[i]
        #logging.info('punto %s =%s', i,  punto)

        # vecinos = np.empty(k_vecinos, object)
        start_time_iter = timer()
        puntos_nube = knn.kmeans_radius_search(n_centroides, punto, vector_training, k_vecinos, metrica,
                                       grupos_capa, puntos_capa, labels_capa, dimensiones, float(radio))

        end_time_iter = timer()
        #logging.info('Index time= %s seconds', end_time_iter - start_time_iter)
        #logging.info('punto %s - time= %s seconds', i, end_time_iter - start_time_iter)


        # Completar el array de vecinos (puntos_nube) con 0s hasta llegar al tamaño de vecinos deseado (k_vecinos)
        # Esto evita el error index out of bounds
        if len(puntos_nube) < k_vecinos:
            puntos_nube = np.append(puntos_nube, np.full((k_vecinos - len(puntos_nube), 3), 0.0, dtype=float), axis=0)

        idx = np.argsort(puntos_nube[:, 1])

        for n in range(k_vecinos):
            # logging.info('id_vecino= %s coords= %s', puntos_nube[idx[n]][0], puntos_nube[idx[n]][2])
            indices[i, n] = puntos_nube[idx[n]][0]
            coords[i, n, :] = puntos_nube[idx[n]][2]

            dists[i, n] = puntos_nube[idx[n]][1]


    end_time_s = timer()
    logging.info('Search time = %s seconds\n', end_time_s - start_time_s)
    logging.info('Average time spent in searching a single point = %s', (end_time_s - start_time_s)/vector_testing.shape[0])
    logging.info('Speed (points/s) = %s\n', vector_testing.shape[0]/(end_time_s - start_time_s))

    # Regarding the knn, method, dataset_name and distance choosen, set the file name to store the neighbors
    file_name = "./experiments/NearestNeighbors/" + dataset + "/knn_" + dataset + "_" + str(k_vecinos) + "_" + metrica + "_" + method + "_tg" + str(tam_grupo) + "_nc" + str(n_centroides) + "_r" + str(radio) + ".hdf5"

    # Store indices, coords and dist into a hdf5 file
    sgn.save_neighbors(indices, coords, dists, file_name)

    logging.info('------------------------------------------------------------------------\n')
