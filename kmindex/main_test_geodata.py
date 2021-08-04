import logging
import kmeans_tree_nparray_geodata as kt


logging.basicConfig(filename='result_kmtnp_testCapacity100-10-2layers_geodata.log', filemode='w', format='%(asctime)s - %(name)s - %(message)s',
                    level=logging.INFO)

# Parámetros de entrada comunes a todas las simulaciones:

# SubText 1
logging.info('------------------------------------------------------------------------')
logging.info(' ')
tam_grupo = 10   #16
n_centroides = 10    #8
logging.info('Sub Text 1, tam_grupo=%s, n_centroides=%s',tam_grupo, n_centroides)
kt.kmeans_tree(tam_grupo,n_centroides)
logging.info(' ')


# SubText 2
logging.info('------------------------------------------------------------------------')
logging.info(' ')
tam_grupo = 10   #16
n_centroides = 9    #8
logging.info('Sub Text 2, tam_grupo=%s, n_centroides=%s',tam_grupo, n_centroides)
kt.kmeans_tree(tam_grupo,n_centroides)
logging.info(' ')


# SubText 3
logging.info('------------------------------------------------------------------------')
logging.info(' ')
tam_grupo = 10   #16
n_centroides = 8    #8
logging.info('Sub Text 3, tam_grupo=%s, n_centroides=%s',tam_grupo, n_centroides)
kt.kmeans_tree(tam_grupo,n_centroides)
logging.info(' ')


# SubText 4
logging.info('------------------------------------------------------------------------')
logging.info(' ')
tam_grupo = 10   #16
n_centroides = 7    #8
logging.info('Sub Text 4, tam_grupo=%s, n_centroides=%s',tam_grupo, n_centroides)
kt.kmeans_tree(tam_grupo,n_centroides)
logging.info(' ')


# SubText 5
logging.info('------------------------------------------------------------------------')
logging.info(' ')
tam_grupo = 10   #16
n_centroides = 6    #8
logging.info('Sub Text 5, tam_grupo=%s, n_centroides=%s',tam_grupo, n_centroides)
kt.kmeans_tree(tam_grupo,n_centroides)
logging.info(' ')


# SubText 6
logging.info('------------------------------------------------------------------------')
logging.info(' ')
tam_grupo = 10   #16
n_centroides = 5    #8
logging.info('Sub Text 6, tam_grupo=%s, n_centroides=%s',tam_grupo, n_centroides)
kt.kmeans_tree(tam_grupo,n_centroides)
logging.info(' ')


# SubText 7
logging.info('------------------------------------------------------------------------')
logging.info(' ')
tam_grupo = 10   #16
n_centroides = 4    #8
logging.info('Sub Text 7, tam_grupo=%s, n_centroides=%s',tam_grupo, n_centroides)
kt.kmeans_tree(tam_grupo,n_centroides)
logging.info(' ')


# SubText 8
logging.info('------------------------------------------------------------------------')
logging.info(' ')
tam_grupo = 10   #16
n_centroides = 3    #8
logging.info('Sub Text 8, tam_grupo=%s, n_centroides=%s',tam_grupo, n_centroides)
kt.kmeans_tree(tam_grupo,n_centroides)
logging.info(' ')


# SubText 9
logging.info('------------------------------------------------------------------------')
logging.info(' ')
tam_grupo = 10   #16
n_centroides = 2    #8
logging.info('Sub Text 9, tam_grupo=%s, n_centroides=%s',tam_grupo, n_centroides)
kt.kmeans_tree(tam_grupo,n_centroides)
logging.info(' ')

'''
# SubText 10
logging.info('------------------------------------------------------------------------')
logging.info(' ')
tam_grupo = 3000   #16
n_centroides = 1500    #8
logging.info('Sub Text 10, tam_grupo=%s, n_centroides=%s',tam_grupo, n_centroides)
kt.kmeans_tree(tam_grupo,n_centroides)
logging.info(' ')


# SubText 11
logging.info('------------------------------------------------------------------------')
logging.info(' ')
tam_grupo = 4000   #16
n_centroides = 2000    #8
logging.info('Sub Text 11, tam_grupo=%s, n_centroides=%s',tam_grupo, n_centroides)
kt.kmeans_tree(tam_grupo,n_centroides)
logging.info(' ')


# SubText 12
logging.info('------------------------------------------------------------------------')
logging.info(' ')
tam_grupo = 5000   #16
n_centroides = 2500    #8
logging.info('Sub Text 12, tam_grupo=%s, n_centroides=%s',tam_grupo, n_centroides)
kt.kmeans_tree(tam_grupo,n_centroides)
logging.info(' ')


# SubText 13
logging.info('------------------------------------------------------------------------')
logging.info(' ')
tam_grupo = 6000   #16
n_centroides = 3000    #8
logging.info('Sub Text 13, tam_grupo=%s, n_centroides=%s',tam_grupo, n_centroides)
kt.kmeans_tree(tam_grupo,n_centroides)
logging.info(' ')'''