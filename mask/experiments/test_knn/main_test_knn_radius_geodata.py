import logging
# import mask.kmeans_tree_npdist_modulado as ktd
import mask.KNN_np as knn
# import mask.data_test as dt
import numpy as np
from timeit import default_timer as timer
import pandas as pd
import other_algorithms.save_get_neighbors as sgn



logging.basicConfig(filename='../../../logs/knn/geodata/euclidean/result_.log',
                    filemode='w', format='%(asctime)s - %(name)s - %(message)s', level=logging.INFO)

# Parámetros de entrada comunes a todas las simulaciones:
tam_grupo = 60    #200 #16
n_centroides = 30 #150 #8

k_vecinos = 15

logging.info('------------------------------------------------------------------------')
logging.info(' ')
logging.info('Text MASK search knn, KNN_np.py')
logging.info('Parametros: ')
logging.info('tam_grupo=%s', tam_grupo)
logging.info('n_centroides=%s', n_centroides)
logging.info('k_vecinos=%s', k_vecinos)


# 1º Leemos los datos
datos = pd.read_csv('../../../data/geo_data/MUNICIPIOS-utf8.csv', sep=';')
datos_geo = pd.DataFrame(datos, columns=['LONGITUD_ETRS89','LATITUD_ETRS89'])
index = datos_geo.index
datos_geo['LONGITUD_ETRS89'] = datos_geo['LONGITUD_ETRS89'].str.replace(',', '.').astype(np.float)
datos_geo['LATITUD_ETRS89'] = datos_geo['LATITUD_ETRS89'].str.replace(',', '.').astype(np.float)

# dt.pinta_geo(datos_geo['LONGITUD_ETRS89'], datos_geo['LATITUD_ETRS89'])

# Separamos los puntos de test y training:
np.random.seed(1234)
datos_geo = datos_geo.to_numpy()
index_testing = np.random.choice(len(datos_geo), 100, replace=False)
vector_testing = datos_geo[index_testing]
index_complete = np.linspace(0, len(datos_geo)-1, len(datos_geo), dtype=int)
index_training = np.setdiff1d(index_complete, index_testing)
vector_training = datos_geo[index_training]

metrica = 'euclidean'
cant_ptos = len(vector_training)
copia_vector_original = datos_geo


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

file_name = '../../../logs/knn/geodata/euclidean/result_kradius15_MUNICIPIOS_tg60_c30.hdf5'
sgn.save_neighbors(indices, coords, dists, file_name)




logging.info(' ')