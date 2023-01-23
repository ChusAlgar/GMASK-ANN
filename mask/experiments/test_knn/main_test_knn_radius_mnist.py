import logging
# import mask.kmeans_tree_npdist_modulado as ktd
import mask.KNN_np as knn
# import mask.data_test as dt
import numpy as np
from timeit import default_timer as timer
import pandas as pd
import other_algorithms.load_train_test_set as lts
import other_algorithms.neighbors_utils as sgn
from pickle import dump, load



logging.basicConfig(filename='./logs/result_kradius5_MNIST_tg1000_c500.log',
                    filemode='w', format='%(asctime)s - %(name)s - %(message)s', level=logging.INFO)

# Parámetros de entrada comunes a todas las simulaciones:
tam_grupo = 1000    #200 #16
n_centroides = 500 #150 #8

k_vecinos = 5

logging.info('------------------------------------------------------------------------')
logging.info(' ')
logging.info('Text MASK search knn, KNN_np.py')
logging.info('Parametros: ')
logging.info('tam_grupo=%s', tam_grupo)
logging.info('n_centroides=%s', n_centroides)
logging.info('k_vecinos=%s', k_vecinos)


# 1º Leemos los datos

# vector_training, vector_testing = lts.load_train_test_h5py(file_name="./data/MNIST_train_test_set.hdf5")
vector_training, vector_testing = lts.load_train_test("MNIST")
metrica = 'euclidean'
cant_ptos = len(vector_training)



# 2º Generamos el árbol
n_capas, grupos_capa, puntos_capa, labels_capa = knn.kmeans_tree(cant_ptos, tam_grupo, n_centroides,
                                                                 metrica, vector_training)

#with open("./other_algorithms/MASK_MNIST" + str(k_vecinos) +".pickle", 'wb') as handle:
#    dump((puntos_capa, labels_capa), handle)


# 3º Buscamos los k vecinos de los puntos de testing
indices = np.empty([len(vector_testing), k_vecinos], dtype=int)
coords = np.empty([len(vector_testing), k_vecinos, 2], dtype=float)
dists = np.empty([len(vector_testing), k_vecinos], dtype=float)
for i in range(len(vector_testing)):
# for i in range(1):
    punto = vector_testing[i]
    logging.info('punto=%s', punto)

    # vecinos = np.empty(k_vecinos, object)
    start_time_iter = timer()
    puntos_nube = knn.kmeans_radius_search(n_centroides, punto, vector_training, k_vecinos, metrica,
                                   grupos_capa, puntos_capa, labels_capa)

    end_time_iter = timer()
    logging.info('punto=%s - time= %s seconds', i, end_time_iter - start_time_iter)


    idx = np.argsort(puntos_nube[:, 1])
    for n in range(k_vecinos):
        logging.info('id_vecino= %s coords= %s', puntos_nube[idx[n]][0], puntos_nube[idx[n]][2])
        indices[i, n] = puntos_nube[idx[n]][0]
        coords[i, n, :] = puntos_nube[idx[n]][2]
        dists[i, n] = puntos_nube[idx[n]][1]

# file_name = '../../../logs/knn/mnist/euclidean/result_kradius5_MNIST_tg60_c30.hdf5'

# Regarding the knn, method, dataset_name and distance choosen, set the file name to store the neighbors
file_name = "./NearestNeighbors/MASK/tg" + str(tam_grupo) + "nc" + str(
        n_centroides) + "/MNIST_euclidean_MASK_" + str(knn) + "nn.hdf5"

#sgn.save_neighbors(indices, coords, dists, file_name)




logging.info(' ')