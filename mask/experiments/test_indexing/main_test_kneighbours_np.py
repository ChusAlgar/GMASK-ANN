import logging
import mask.kmeans_tree_npdist_modulado as ktd
import mask.kneighbours_np as kn
import mask.data_test as dt
import numpy as np
from timeit import default_timer as timer



logging.basicConfig(filename='../../../experiments/logs/kneighbours/result_iter3k3_tg16_c8_npc200.log', filemode='w', format='%(asctime)s - %(name)s - %(message)s',
                    level=logging.INFO)

# Parámetros de entrada comunes a todas las simulaciones:
nclouds = 8
tam_grupo = 16
n_centroides = 8
npc = 200
overlap = True

k_vecinos = 3
punto = np.array([[15, 17.5]], np.float)

logging.info('------------------------------------------------------------------------')
logging.info(' ')
logging.info('Text MASK search k_neighbours, kneighbours_np.py')
logging.info('Parametros: ')
logging.info('nclouds=%s', nclouds)
logging.info('tam_grupo=%s', tam_grupo)
logging.info('n_centroides=%s', n_centroides)
logging.info('npc=%s', npc)
logging.info('overlap=%s', overlap)

# 1º Generamos los datos
vector_original, coordx, coordy, puntos_nube = dt.generate_data_gaussian_clouds(nclouds, npc, overlap)

metrica = 'euclidean'
cant_ptos = nclouds * npc
copia_vector_original = vector_original

start_time_iter = timer()
n_capas, grupos_capa, puntos_capa, labels_capa = kn.kmeans_treeini(cant_ptos, tam_grupo, n_centroides,
                                                                    metrica, copia_vector_original)
# Pinto las nubes de puntos originales junto con los centroides sacados
dt.pinta(coordx, coordy, puntos_capa[n_capas-1], npc, nclouds)
aciertos, fallos, vecordenacion = kn.kmeans_searchini(n_capas, n_centroides, np.array(copia_vector_original), tam_grupo, metrica,
                                                       grupos_capa, puntos_capa, labels_capa)
# cont = 0
vector_puntos = []
for j in range(len(vecordenacion)):
    indices = vecordenacion[j, 1]
    for indice in indices:
        # copia_vector_original[cont] = vector_original[indice]
        vector_puntos.append(vector_original[indice])
        # cont += 1

end_time_iter = timer()
logging.info('iter=0 - time= %s seconds', end_time_iter - start_time_iter)
logging.info('iter=0 - aciertos= %s - fallos= %s', aciertos, fallos)

for itera in range(3):
    start_time_iter = timer()
    n_capas, grupos_capa, puntos_capa, labels_capa = kn.kmeans_tree(tam_grupo, n_centroides, metrica,
                                                                    vector_puntos, vecordenacion)
    #                                                                  copia_vector_original, vecordenacion)
    # Pinto las nubes de puntos originales junto con los centroides sacados
    dt.pinta(coordx, coordy, puntos_capa[n_capas-1], npc, nclouds)

    # copia_vector_original = np.array(copia_vector_original)
    vector_puntos = np.array(vector_puntos)

    # vecino = kn.kneighsbours_search(1, punto, n_capas, n_centroides, copia_vector_original, puntos_nube, metrica,
    #                                grupos_capa, puntos_capa, labels_capa)
    # aciertos, fallos, vecordenacion = kn.kmeans_search(n_capas, n_centroides, copia_vector_original, tam_grupo, metrica,
    aciertos, fallos, vecordenacion = kn.kmeans_search(n_capas, np.array(vector_original), vecordenacion, tam_grupo,
                                                       metrica, grupos_capa, puntos_capa, labels_capa)

    # cont = 0
    vector_puntos = []
    for j in range(len(vecordenacion)):
       indices = vecordenacion[j, 1]
       for indice in indices:
           # copia_vector_original[cont] = vector_original[indice]
           vector_puntos.append(vector_original[indice])
    vector_puntos = np.array(vector_puntos)

    end_time_iter = timer()
    logging.info('iter=%s - time= %s seconds', itera+1, end_time_iter - start_time_iter)
    logging.info('iter=%s - aciertos= %s - fallos= %s', itera+1, aciertos, fallos)


    # logging.info('fin iter= %s, vecino= %s', i, vecino[0])
    # idvecino = vecino[0]
    # idvecino = int(idvecino)
    # if idvecino == 0:
    #     copia_vector_original = copia_vector_original[1:]
    # elif idvecino == cant_ptos-1:
    #     copia_vector_original = copia_vector_original[:cant_ptos-2]
    # else:
    #     trozo1 = copia_vector_original[0:idvecino]
    #     trozo2 = copia_vector_original[(idvecino+1):]
    #     copia_vector_original = np.concatenate([trozo1, trozo2])
    # cant_ptos -= 1

# vecinos = kn.kneighbours_search(k_vecinos, punto, n_capas, n_centroides, copia_vector_original, metrica, grupos_capa, puntos_capa, labels_capa)
vecinos = kn.kneighbours_search(k_vecinos, punto, n_capas, vecordenacion, np.array(vector_original), metrica, grupos_capa, puntos_capa, labels_capa)

# dt.pinta_grupos(vector_original, npc, nclouds, tam_grupo)
# dt.pinta_vecinos(vector_original, punto[0], vecinos, npc, nclouds, tam_grupo)

logging.info('Indices vecinos')
for ele in vecinos:
    logging.info('vecino= %s', ele[0])

# # 2º Construimos el árbol
# metrica = 'euclidean'
# cant_ptos = nclouds * npc
# n_capas, grupos_capa, puntos_capa, labels_capa = ktd.kmeans_tree(cant_ptos, tam_grupo, n_centroides, metrica, vector_original)
#
# # Pinto las nubes de puntos originales junto con los centroides sacados
# dt.pinta(coordx, coordy, puntos_capa[n_capas-1], npc, nclouds)
# # dt.pinta_info_nube(coordx, coordy, puntos_capa, grupos_capa, labels_capa)
#
# # 3º Hacemos la búsqueda de los k_vecinos
# vector_original = np.array(vector_original)
#
# if k_vecinos <= tam_grupo:
#     vecinos = kn.kneighbours_search(k_vecinos, punto, n_capas, n_centroides, vector_original, puntos_nube, metrica, grupos_capa, puntos_capa, labels_capa)
#
#     #dt.pinta_grupos(vector_original, npc, nclouds, tam_grupo)
#     #dt.pinta_vecinos(vector_original, punto[0], vecinos, npc, nclouds, tam_grupo)
#
#     logging.info('Indices vecinos')
#     for ele in vecinos:
#         logging.info('vecino= %s', ele[0])
#     #print("Porcentaje de aciertos en la iteración ", iter, ": ", aciertos * 100 / (npc * nclouds))
#     #print("Porcentaje de fallos en la iteración ", iter, ": ", fallos * 100 / (npc * nclouds))
#     #logging.info('Porcentaje de aciertos= %s', aciertos * 100 / (npc * nclouds))
#     #logging.info('Porcentaje de fallos= %s', fallos * 100 / (npc * nclouds))

logging.info(' ')