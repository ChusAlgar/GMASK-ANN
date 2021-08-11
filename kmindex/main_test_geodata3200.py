import logging
import kmeans_tree_nparray_geodata as kt


logging.basicConfig(filename='result_kmtnp_testCapacity-3200-2layers_geodata.log', filemode='w', format='%(asctime)s - %(name)s - %(message)s',
                    level=logging.INFO)

# Parámetros de entrada comunes a todas las simulaciones:

# SubText 1
logging.info('------------------------------------------------------------------------')
logging.info(' ')
tam_grupo = 3200   #16
n_centroides = 3200    #8
logging.info('Sub Text 1, tam_grupo=%s, n_centroides=%s',tam_grupo, n_centroides)
kt.kmeans_tree(tam_grupo,n_centroides)
logging.info(' ')


# SubText 2
logging.info('------------------------------------------------------------------------')
logging.info(' ')
tam_grupo = 3200   #16
n_centroides = 2880    #8
logging.info('Sub Text 2, tam_grupo=%s, n_centroides=%s',tam_grupo, n_centroides)
kt.kmeans_tree(tam_grupo,n_centroides)
logging.info(' ')


# SubText 3
logging.info('------------------------------------------------------------------------')
logging.info(' ')
tam_grupo = 3200   #16
n_centroides = 2560    #8
logging.info('Sub Text 3, tam_grupo=%s, n_centroides=%s',tam_grupo, n_centroides)
kt.kmeans_tree(tam_grupo,n_centroides)
logging.info(' ')


# SubText 4
logging.info('------------------------------------------------------------------------')
logging.info(' ')
tam_grupo = 3200   #16
n_centroides = 2240    #8
logging.info('Sub Text 4, tam_grupo=%s, n_centroides=%s',tam_grupo, n_centroides)
kt.kmeans_tree(tam_grupo,n_centroides)
logging.info(' ')


# SubText 5
logging.info('------------------------------------------------------------------------')
logging.info(' ')
tam_grupo = 3200   #16
n_centroides = 1920    #8
logging.info('Sub Text 5, tam_grupo=%s, n_centroides=%s',tam_grupo, n_centroides)
kt.kmeans_tree(tam_grupo,n_centroides)
logging.info(' ')


# SubText 6
logging.info('------------------------------------------------------------------------')
logging.info(' ')
tam_grupo = 3200   #16
n_centroides = 1600    #8
logging.info('Sub Text 6, tam_grupo=%s, n_centroides=%s',tam_grupo, n_centroides)
kt.kmeans_tree(tam_grupo,n_centroides)
logging.info(' ')


# SubText 7
logging.info('------------------------------------------------------------------------')
logging.info(' ')
tam_grupo = 3200   #16
n_centroides = 1280    #8
logging.info('Sub Text 7, tam_grupo=%s, n_centroides=%s',tam_grupo, n_centroides)
kt.kmeans_tree(tam_grupo,n_centroides)
logging.info(' ')


# SubText 8
logging.info('------------------------------------------------------------------------')
logging.info(' ')
tam_grupo = 3200   #16
n_centroides = 960    #8
logging.info('Sub Text 8, tam_grupo=%s, n_centroides=%s',tam_grupo, n_centroides)
kt.kmeans_tree(tam_grupo,n_centroides)
logging.info(' ')


# SubText 9
logging.info('------------------------------------------------------------------------')
logging.info(' ')
tam_grupo = 3200   #16
n_centroides = 640    #8
logging.info('Sub Text 9, tam_grupo=%s, n_centroides=%s',tam_grupo, n_centroides)
kt.kmeans_tree(tam_grupo,n_centroides)
logging.info(' ')