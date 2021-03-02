import logging
import kmeans_tree as kt


logging.basicConfig(filename='result_tgncNoverlap.log', filemode='w', format='%(asctime)s - %(name)s - %(message)s',
                    level=logging.INFO)

# Parámetros de entrada comunes a todas las simulaciones:
nclouds = 8
npc = 10000
overlap = False


#SubText 1
tam_grupo = 10
n_centroides = 5
logging.info('------------------------------------------------------------------------')
logging.info(' ')
logging.info('Sub Text 1, tam_grupo=%s  n_centroides=%s', tam_grupo, n_centroides)
kt.kmeans_tree(nclouds,npc,tam_grupo,n_centroides,overlap)
logging.info(' ')

#SubText 2
tam_grupo = 30
n_centroides = 15
logging.info('------------------------------------------------------------------------')
logging.info(' ')
logging.info('Sub Text 2, tam_grupo=%s  n_centroides=%s', tam_grupo, n_centroides)
kt.kmeans_tree(nclouds,npc,tam_grupo,n_centroides,overlap)
logging.info(' ')

#SubText 3
tam_grupo = 50
n_centroides = 25
logging.info('------------------------------------------------------------------------')
logging.info(' ')
logging.info('Sub Text 3, tam_grupo=%s  n_centroides=%s', tam_grupo, n_centroides)
kt.kmeans_tree(nclouds,npc,tam_grupo,n_centroides,overlap)
logging.info(' ')


#SubText 4
tam_grupo = 70
n_centroides = 35
logging.info('------------------------------------------------------------------------')
logging.info(' ')
logging.info('Sub Text 4, tam_grupo=%s  n_centroides=%s', tam_grupo, n_centroides)
kt.kmeans_tree(nclouds,npc,tam_grupo,n_centroides,overlap)
logging.info(' ')


#SubText 5
tam_grupo = 90
n_centroides = 45
logging.info('------------------------------------------------------------------------')
logging.info(' ')
logging.info('Sub Text 5, tam_grupo=%s  n_centroides=%s', tam_grupo, n_centroides)
kt.kmeans_tree(nclouds,npc,tam_grupo,n_centroides,overlap)
logging.info(' ')


#SubText 6
tam_grupo = 110
n_centroides = 55
logging.info('------------------------------------------------------------------------')
logging.info(' ')
logging.info('Sub Text 6, tam_grupo=%s  n_centroides=%s', tam_grupo, n_centroides)
kt.kmeans_tree(nclouds,npc,tam_grupo,n_centroides,overlap)
logging.info(' ')