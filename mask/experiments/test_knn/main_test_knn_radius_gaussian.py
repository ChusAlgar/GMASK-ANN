import logging
# import mask.kmeans_tree_npdist_modulado as ktd
import mask.KNN_np as knn
import mask.data_test as dt
import numpy as np
from timeit import default_timer as timer
import pandas as pd
import other_algorithms.save_get_neighbors as sgn



logging.basicConfig(filename='../../../logs/knn/gaussian/euclidean/result_k5_tg1000_c500_npc100000.log',
                    filemode='w', format='%(asctime)s - %(name)s - %(message)s', level=logging.INFO)

# Parámetros de entrada comunes a todas las simulaciones:
nclouds = 8
tam_grupo = 1000  # 16
n_centroides = 500  # 8
npc = 100000
overlap = True

k_vecinos = 5
# punto = np.array([[15, 17.5]], np.float)

logging.info('------------------------------------------------------------------------')
logging.info(' ')
logging.info('Text MASK search knn, KNN_np.py')
logging.info('Parametros: ')
logging.info('nclouds=%s', nclouds)
logging.info('tam_grupo=%s', tam_grupo)
logging.info('n_centroides=%s', n_centroides)
logging.info('npc=%s', npc)
logging.info('overlap=%s', overlap)
# logging.info('punto=%s', punto)
logging.info('k_vecinos=%s', k_vecinos)


# 1º Generamos los datos
vector_original, coordx, coordy, puntos_nube = dt.generate_data_gaussian_clouds(nclouds, npc, overlap)

# Separamos los puntos de test y training:
vector_original = np.array(vector_original)
index_testing = np.random.choice(len(vector_original), 100, replace=False)
vector_testing = vector_original[index_testing]
index_complete = np.linspace(0, len(vector_original)-1, len(vector_original), dtype=int)
index_training = np.setdiff1d(index_complete, index_testing)
vector_training = vector_original[index_training]

metrica = 'euclidean'
# cant_ptos = nclouds * npc
cant_ptos = len(vector_training)
copia_vector_original = vector_original


# 2º Generamos el árbol
n_capas, grupos_capa, puntos_capa, labels_capa = knn.kmeans_tree(cant_ptos, tam_grupo, n_centroides,
#                                                                   metrica, copia_vector_original)
                                                                 metrica, vector_training)


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
    # if len(puntos_nube) < k_vecinos:
    #     indices[i] = np.empty(len(puntos_nube), object)
    #     coords[i] = np.empty(len(puntos_nube), object)
    #     dists[i] = np.empty(len(puntos_nube), object)
    #     for n in range(len(puntos_nube)):
    #         logging.info('id_vecino= %s coords= %s', puntos_nube[idx[n]][0], puntos_nube[idx[n]][2])
    #         indices[i][n] = puntos_nube[idx[n]][0]
    #         coords[i][n] = puntos_nube[idx[n]][2]
    #         dists[i][n] = puntos_nube[idx[n]][1]
    # else:
    #     indices[i] = np.empty(k_vecinos, object)
    #     coords[i] = np.empty(k_vecinos, object)
    #     dists[i] = np.empty(k_vecinos, object)
    for n in range(k_vecinos):
        logging.info('id_vecino= %s coords= %s', puntos_nube[idx[n]][0], puntos_nube[idx[n]][2])
        indices[i, n] = puntos_nube[idx[n]][0]
        coords[i, n, :] = puntos_nube[idx[n]][2]
        dists[i, n] = puntos_nube[idx[n]][1]

file_name = '../../../logs/knn/gaussian/euclidean/result_kradius5_gaussian8x200_tg60_c30.hdf5'
sgn.save_neighbors(indices, coords, dists, file_name)




logging.info(' ')