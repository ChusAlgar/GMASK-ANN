import logging
import kmeans_tree as kt


logging.basicConfig(filename='result_rel9072_tgncOverlap.log', filemode='w', format='%(asctime)s - %(name)s - %(message)s',
                    level=logging.INFO)

# Parámetros de entrada comunes a todas las simulaciones:
nclouds = 8
npc = 10000
overlap = True


# SubText 90_72
tam_grupo = 90
n_centroides = 72
logging.info('------------------------------------------------------------------------')
logging.info(' ')
logging.info('Sub Text 90_72, tam_grupo=%s  n_centroides=%s', tam_grupo, n_centroides)
kt.kmeans_tree(nclouds,npc,tam_grupo,n_centroides,overlap)
logging.info(' ')