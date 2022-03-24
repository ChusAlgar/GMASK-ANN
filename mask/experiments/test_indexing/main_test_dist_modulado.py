import logging
import mask.kmeans_tree_nparray as kt
import mask.kmeans_tree_npdist_modulado as ktd
import mask.data_test as dt
import numpy as np



logging.basicConfig(filename='../../../logs/moduladov2/manhattan/result_km_overlap2.log', filemode='w', format='%(asctime)s - %(name)s - %(message)s',
                    level=logging.INFO)

# Parámetros de entrada comunes a todas las simulaciones:
nclouds = 8
tam_grupo = 16
n_centroides = 8
npc = 200
overlap = True


# SubText 1_kt
logging.info('------------------------------------------------------------------------')
logging.info(' ')
logging.info('Sub Text MASK, kmeans_tree_nparray')
kt.kmeans_tree(nclouds,npc,tam_grupo,n_centroides,overlap)
logging.info(' ')


# SubText 2_MASKmodulado
logging.info('------------------------------------------------------------------------')
logging.info(' ')
logging.info('Sub Text MASK modulado con Numba, kmeans_tree_npdist_modulado')

# 1º Generamos los datos
vector_original, coordx, coordy, puntos_nube = dt.generate_data_gaussian_clouds(nclouds, npc, overlap)

# 2º Construimos el árbol
metrica = 'manhattan'
cant_ptos = nclouds * npc
n_capas, grupos_capa, puntos_capa, labels_capa = ktd.kmeans_tree(cant_ptos, tam_grupo, n_centroides, metrica, vector_original)

# Pinto las nubes de puntos originales junto con los centroides sacados
#dt.pinta(coordx, coordy, puntos_capa[n_capas], npc, nclouds)
# dt.pinta_info_nube(coordx, coordy, puntos_capa, grupos_capa, labels_capa)

# 3º Hacemos la búsqueda
vector_original = np.array(vector_original)

aciertos, fallos = ktd.kmeans_search(n_capas, n_centroides, vector_original, puntos_nube, metrica, grupos_capa, puntos_capa, labels_capa)

print("Porcentaje de aciertos en la iteración ", iter, ": ", aciertos * 100 / (npc * nclouds))
print("Porcentaje de fallos en la iteración ", iter, ": ", fallos * 100 / (npc * nclouds))
logging.info('Porcentaje de aciertos= %s', aciertos * 100 / (npc * nclouds))
logging.info('Porcentaje de fallos= %s', fallos * 100 / (npc * nclouds))

logging.info(' ')