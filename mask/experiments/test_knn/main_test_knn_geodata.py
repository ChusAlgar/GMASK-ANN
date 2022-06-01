import logging
import mask.kmeans_tree_npdist_modulado as ktd
import mask.KNN_np as knn
import mask.data_test as dt
import numpy as np
from timeit import default_timer as timer
import pandas as pd



logging.basicConfig(filename='../../../logs/knn/geodata/euclidean/result_k5_MUNICIPIOSA_tg60_c30.log',
                    filemode='w', format='%(asctime)s - %(name)s - %(message)s', level=logging.INFO)

# Parámetros de entrada comunes a todas las simulaciones:
# nclouds = 8
tam_grupo = 60    #200 #16
n_centroides = 30 #150 #8
# npc = 100000
# overlap = True

k_vecinos = 5
# punto = np.array([[15, 17.5]], np.float)

logging.info('------------------------------------------------------------------------')
logging.info(' ')
logging.info('Text MASK search knn, KNN_np.py')
logging.info('Parametros: ')
# logging.info('nclouds=%s', nclouds)
logging.info('tam_grupo=%s', tam_grupo)
logging.info('n_centroides=%s', n_centroides)
# logging.info('npc=%s', npc)
# logging.info('overlap=%s', overlap)
# logging.info('punto=%s', punto)
logging.info('k_vecinos=%s', k_vecinos)


# 1º Leemos los datos
datos = pd.read_csv('../../../data/geo_data/MUNICIPIOS-utf8.csv', sep=';')
datos_geo = pd.DataFrame(datos, columns=['LONGITUD_ETRS89','LATITUD_ETRS89'])
index = datos_geo.index
# cant_ptos = len(index)
datos_geo['LONGITUD_ETRS89'] = datos_geo['LONGITUD_ETRS89'].str.replace(',', '.').astype(np.float)
datos_geo['LATITUD_ETRS89'] = datos_geo['LATITUD_ETRS89'].str.replace(',', '.').astype(np.float)

# dt.pinta_geo(datos_geo['LONGITUD_ETRS89'], datos_geo['LATITUD_ETRS89'])

# Separamos los puntos de test y training:
np.random.seed(1234)
# vector_original = datos_geo
datos_geo = datos_geo.to_numpy()
index_testing = np.random.choice(len(datos_geo), 100, replace=False)
vector_testing = datos_geo[index_testing]
index_complete = np.linspace(0, len(datos_geo)-1, len(datos_geo), dtype=int)
index_training = np.setdiff1d(index_complete, index_testing)
vector_training = datos_geo[index_training]

metrica = 'euclidean'
# cant_ptos = nclouds * npc
cant_ptos = len(vector_training)
copia_vector_original = datos_geo


# 2º Generamos el árbol
n_capas, grupos_capa, puntos_capa, labels_capa = knn.kmeans_tree(cant_ptos, tam_grupo, n_centroides,
#                                                                   metrica, copia_vector_original)
                                                                 metrica, vector_training)


# 3º Buscamos los k vecinos de los puntos de testing
for i in range(len(vector_testing)):
    punto = vector_testing[i]
    logging.info('punto=%s', punto)

    vecinos = np.empty(k_vecinos, object)
    centroides_examinados = np.empty(len(grupos_capa[0]), object)
    for i in range(len(grupos_capa[0])):
        centroides_examinados[i] = np.zeros(len(puntos_capa[0][i]), dtype=int)
    n = 0
    # for n in range(k_vecinos):
    while n < k_vecinos:
        start_time_iter = timer()
        # almacenado = knn.kmeans_search(n_capas, n_centroides, punto, np.array(vector_original), vecinos,
        almacenado = knn.kmeans_search(n_capas, n_centroides, punto, vector_training, vecinos,
                                       centroides_examinados, n, metrica, grupos_capa, puntos_capa, labels_capa)

        end_time_iter = timer()
        logging.info('iter=%s - time= %s seconds', n, end_time_iter - start_time_iter)
        # En las siguientes iteraciones buscamos el vecino del vecino
        if almacenado:
            punto = vecinos[n][2]
            punto = np.reshape(punto, (1, 2))
            logging.info('iter=%s - id_vecino= %s', n, vecinos[n][0])
            n += 1


logging.info(' ')