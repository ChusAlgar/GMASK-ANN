from experiments.neighbors_utils import *
import logging
import pandas as pd

# Set var for experiments:
#datasets = ['wdbc', 'municipios', 'MNIST', 'NYtimes', 'GLOVE']
datasets = ['GLOVE']
distances = ['manhattan', 'euclidean', 'chebyshev']
methods = ['FLANN', 'PYNN', 'MASK', 'GMASK']
baseline = 'KDTree'
knn = [5, 10, 15]

mask_algorithm = 'kmeans'
mask_implementation = 'kclust'

gmask_algorithm = 'kmedoids'
gmask_implementation = 'fastkmedoids'


def benchmark_error_rate():

    for da in datasets:

        # MASK & GMASK configurations
        if da == 'NYtimes':
            tg = 1000
            nc = 500
            r = 30

        elif da == 'GLOVE':
            tg = 1000
            nc = 500
            r = 250

        elif da == 'MNIST':
            tg = 1000
            nc = 500
            r = 80000

        elif da == 'municipios':
            tg = 60
            nc = 30
            r = 40

        elif da == 'wdbc':
            tg = 50
            nc = 25
            r = 7500

        # Set logging info
        logging.basicConfig(
            filename='./experiments/logs/' + da + '/' + da + "_tg" + str(tg) + "_nc" + str(nc) + "_r" + str(r) + "_" + str(mask_algorithm) + "" + str(mask_implementation) + "_" + str(gmask_algorithm) + "" + str(gmask_implementation) + '_errorrate.log',
            filemode='w', format='%(asctime)s - %(name)s - %(message)s', level=logging.INFO)
        logging.info('------------------------------------------------------------------------')
        logging.info('                       KNN over %s Dataset ERROR RATE', da)
        logging.info('------------------------------------------------------------------------')
        logging.info('Search of k neighbors over a choosen dataset, using different methods, benchmark.py')
        logging.info('------------------------------------------------------------------------\n')

        logging.info('Distances: %s ', distances)
        logging.info('Methods: %s', methods)
        logging.info('KNN: %s', knn)
        logging.info('MASK params: tg=%s, nc=%s, r=%s, algorithm=%s, implementation=%s\n', tg, nc, r, mask_algorithm, mask_implementation)
        logging.info('GMASK params: tg=%s, nc=%s, r=%s, algorithm=%s, implementation=%s\n', tg, nc, r, gmask_algorithm, gmask_implementation)

        # From a chosen dataset, calculate recalls, store them and print graph
        da_error_rate = []
        for di in distances:
            logging.info('------------  %s distance  --------------------', di)
            di_error_rate = []
            for method in methods:
                m_error_rate = []

                logging.info('')
                logging.info('-- %s method --', method)
                logging.info('')

                for k in knn:

                    file_name_le = "./experiments/NearestNeighbors/" + da + "/knn_" + da + "_" + str(k) + "_" + di + "_" + baseline + ".hdf5"

                    if method == 'MASK':
                        file_name = "./experiments/NearestNeighbors/" + da + "/knn_" + da + "_" + str(
                            k) + "_" + di + "_" + method + "_tg" + str(tg) + "_nc" + str(nc) + "_r" + str(
                            r) + "_" + str(mask_algorithm) + "_" + str(mask_implementation) + ".hdf5"

                    elif method == 'GMASK':
                        file_name = "./experiments/NearestNeighbors/" + da + "/knn_" + da + "_" + str(
                            k) + "_" + di + "_" + method + "_tg" + str(tg) + "_nc" + str(nc) + "_r" + str(
                            r) + "_" + str(gmask_algorithm) + "_" + str(gmask_implementation) + ".hdf5"

                    else:
                        file_name = "./experiments/NearestNeighbors/" + da + "/knn_" + da + "_" + str(k) + "_" + di + "_" + method + ".hdf5"

                    er = error_rate(da, di, method, k, False, file_name_le, file_name)

                    m_error_rate.append(er)

                di_error_rate.append(m_error_rate)
            da_error_rate.append(di_error_rate)

def benchmark_recall():

    recalls = pd.DataFrame(columns=['Dataset', 'k', 'Distance', 'Method', 'Recall'])

    for da in datasets:

        # MASK & GMASK configurations
        if da == 'NYtimes':
            tg = 1000
            nc = 500
            r = 30

        elif da == 'GLOVE':
            tg = 1000
            nc = 500
            r = 250

        elif da == 'MNIST':
            tg = 1000
            nc = 500
            r = 80000

        elif da == 'municipios':
            tg = 60
            nc = 30
            r = 40

        elif da == 'wdbc':
            tg = 50
            nc = 25
            r = 7500

        # Set logging info
        logging.basicConfig(filename='./experiments/logs/' + da + '/' + da + "_tg" + str(tg) + "_nc" + str(nc) + "_r" + str(r) + "_" + "MASK" + "." + str(mask_implementation) + "_" + "GMASK" + "." + str(gmask_implementation) +'_recall.log',
                            filemode='w', format='%(asctime)s - %(name)s - %(message)s', level=logging.INFO)
        logging.info('------------------------------------------------------------------------')
        logging.info('                       KNN over %s Dataset RECALL', da)
        logging.info('------------------------------------------------------------------------')
        logging.info('Search of k neighbors over a choosen dataset, using different methods, benchmark.py')
        logging.info('------------------------------------------------------------------------\n')

        logging.info('Distances: %s ', distances)
        logging.info('Methods: %s', methods)
        logging.info('KNN: %s', knn)
        logging.info('MASK params: tg=%s, nc=%s, r=%s, algorithm=%s, implementation=%s', tg, nc, r, mask_algorithm, mask_implementation)
        logging.info('GMASK params: tg=%s, nc=%s, r=%s, algorithm=%s, implementation=%s\n', tg, nc, r, gmask_algorithm, gmask_implementation)

        # From a chosen dataset, calculate recalls, store them and print graph
        for di in distances:
            logging.info('------------  %s distance  --------------------\n', di)
            for method in methods:

                logging.info('-- %s method --\n', method)

                for k in knn:

                    file_name_le = "./experiments/NearestNeighbors/" + da + "/knn_" + da + "_" + str(
                        k) + "_" + di + "_" + baseline + ".hdf5"

                    if method == 'MASK':
                        file_name = "./experiments/NearestNeighbors/" + da + "/knn_" + da + "_" + str(
                            k) + "_" + di + "_" + method + "_tg" + str(tg) + "_nc" + str(nc) + "_r" + str(r) + "_" + str(mask_algorithm) + "_" + str(mask_implementation) + ".hdf5"

                    elif method == 'GMASK':
                        file_name = "./experiments/NearestNeighbors/" + da + "/knn_" + da + "_" + str(
                            k) + "_" + di + "_" + method + "_tg" + str(tg) + "_nc" + str(nc) + "_r" + str(r) + "_" + str(gmask_algorithm) + "_" + str(gmask_implementation) + ".hdf5"
                    else:
                        file_name = "./experiments/NearestNeighbors/" + da + "/knn_" + da + "_" + str(
                            k) + "_" + di + "_" + method + ".hdf5"

                    if not os.path.isfile(file_name):
                        re = np.nan
                    else:
                        re = recall(da, di, method, k, False, file_name_le, file_name)

                    recalls = pd.concat([recalls, pd.DataFrame([{'Dataset': da, 'k': k, 'Distance': di, 'Method': method, 'Recall': re}])], ignore_index=True)


    # Show results on a graph
    #print_compare_recall_graph(recalls)
    print_recall_heatmap(datasets, distances, methods, knn, recalls)
    #print(da_recalls)

benchmark_recall()

exit(0)
