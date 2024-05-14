import logging
import kmeans_tree_nparray as kt


logging.basicConfig(filename='result_kmtnp_rel_tgncOverlap.log', filemode='w', format='%(asctime)s - %(name)s - %(message)s',
                    level=logging.INFO)

# Parámetros de entrada comunes a todas las simulaciones:
nclouds = 8
npc = 10000
overlap = True


# SubText 10_2
tam_grupo = 10
n_centroides = 2
logging.info('------------------------------------------------------------------------')
logging.info(' ')
logging.info('Sub Text 10_2, tam_grupo=%s  n_centroides=%s', tam_grupo, n_centroides)
kt.kmeans_tree(nclouds,npc,tam_grupo,n_centroides,overlap)
logging.info(' ')


# SubText 10_5
tam_grupo = 10
n_centroides = 5
logging.info('------------------------------------------------------------------------')
logging.info(' ')
logging.info('Sub Text 10_5, tam_grupo=%s  n_centroides=%s', tam_grupo, n_centroides)
kt.kmeans_tree(nclouds,npc,tam_grupo,n_centroides,overlap)
logging.info(' ')


# SubText 10_8
tam_grupo = 10
n_centroides = 8
logging.info('------------------------------------------------------------------------')
logging.info(' ')
logging.info('Sub Text 10_8, tam_grupo=%s  n_centroides=%s', tam_grupo, n_centroides)
kt.kmeans_tree(nclouds,npc,tam_grupo,n_centroides,overlap)
logging.info(' ')


logging.info('------------------------------------------------------------------------')
# SubText 50_10
tam_grupo = 50
n_centroides = 10
logging.info('------------------------------------------------------------------------')
logging.info(' ')
logging.info('Sub Text 50_10, tam_grupo=%s  n_centroides=%s', tam_grupo, n_centroides)
kt.kmeans_tree(nclouds,npc,tam_grupo,n_centroides,overlap)
logging.info(' ')


# SubText 50_25
tam_grupo = 50
n_centroides = 25
logging.info('------------------------------------------------------------------------')
logging.info(' ')
logging.info('Sub Text 50_25, tam_grupo=%s  n_centroides=%s', tam_grupo, n_centroides)
kt.kmeans_tree(nclouds,npc,tam_grupo,n_centroides,overlap)
logging.info(' ')


# SubText 50_40
tam_grupo = 50
n_centroides = 40
logging.info('------------------------------------------------------------------------')
logging.info(' ')
logging.info('Sub Text 50_40, tam_grupo=%s  n_centroides=%s', tam_grupo, n_centroides)
kt.kmeans_tree(nclouds,npc,tam_grupo,n_centroides,overlap)
logging.info(' ')


logging.info('------------------------------------------------------------------------')
# SubText 90_18
tam_grupo = 90
n_centroides = 18
logging.info('------------------------------------------------------------------------')
logging.info(' ')
logging.info('Sub Text 90_18, tam_grupo=%s  n_centroides=%s', tam_grupo, n_centroides)
kt.kmeans_tree(nclouds,npc,tam_grupo,n_centroides,overlap)
logging.info(' ')


# SubText 90_45
tam_grupo = 90
n_centroides = 45
logging.info('------------------------------------------------------------------------')
logging.info(' ')
logging.info('Sub Text 90_45, tam_grupo=%s  n_centroides=%s', tam_grupo, n_centroides)
kt.kmeans_tree(nclouds,npc,tam_grupo,n_centroides,overlap)
logging.info(' ')


# SubText 90_72
tam_grupo = 90
n_centroides = 72
logging.info('------------------------------------------------------------------------')
logging.info(' ')
logging.info('Sub Text 90_72, tam_grupo=%s  n_centroides=%s', tam_grupo, n_centroides)
kt.kmeans_tree(nclouds,npc,tam_grupo,n_centroides,overlap)
logging.info(' ')