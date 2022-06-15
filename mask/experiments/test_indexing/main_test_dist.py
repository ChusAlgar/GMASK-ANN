import logging
import mask.kmeans_tree_nparray as kt
import mask.kmeans_tree_np_dist as ktd


logging.basicConfig(filename='../../../logs/dist/Manhattan/result_km_dist4.log', filemode='w', format='%(asctime)s - %(name)s - %(message)s',
                    level=logging.INFO)

# Parámetros de entrada comunes a todas las simulaciones:
nclouds = 8
tam_grupo = 16
n_centroides = 8
npc = 200
overlap = False


# SubText 1_kt
logging.info('------------------------------------------------------------------------')
logging.info(' ')
logging.info('Sub Text 1, kt')
kt.kmeans_tree(nclouds,npc,tam_grupo,n_centroides,overlap)
logging.info(' ')


# SubText 2_ktd
logging.info('------------------------------------------------------------------------')
logging.info(' ')
logging.info('Sub Text 2, ktd')
ktd.kmeans_tree(nclouds,npc,tam_grupo,n_centroides,overlap)
logging.info(' ')