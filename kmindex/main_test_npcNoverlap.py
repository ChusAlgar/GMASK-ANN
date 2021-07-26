import logging
# import error_kmeans_tree as kt
import kmeans_tree_nparray as kt


logging.basicConfig(filename='result_npcNoverlap.log', filemode='w', format='%(asctime)s - %(name)s - %(message)s',
                    level=logging.INFO)

# Parámetros de entrada comunes a todas las simulaciones:
nclouds = 8
tam_grupo = 16
n_centroides = 8
overlap = False


# SubText 1_npc=200
npc = 200
logging.info('------------------------------------------------------------------------')
logging.info(' ')
logging.info('Sub Text 1, npc=%s',npc)
kt.kmeans_tree(nclouds,npc,tam_grupo,n_centroides,overlap)
logging.info(' ')


# SubText 2_npc=1000
npc = 1000
logging.info('------------------------------------------------------------------------')
logging.info(' ')
logging.info('Sub Text 2, npc=%s',npc)
kt.kmeans_tree(nclouds,npc,tam_grupo,n_centroides,overlap)
logging.info(' ')


# SubText 3_npc=1000
npc = 10000
logging.info('------------------------------------------------------------------------')
logging.info(' ')
logging.info('Sub Text 3, npc=%s',npc)
kt.kmeans_tree(nclouds,npc,tam_grupo,n_centroides,overlap)
logging.info(' ')


# SubText 4_npc=10000
npc = 100000
logging.info('------------------------------------------------------------------------')
logging.info(' ')
logging.info('Sub Text 4, npc=%s',npc)
kt.kmeans_tree(nclouds,npc,tam_grupo,n_centroides,overlap)
logging.info(' ')