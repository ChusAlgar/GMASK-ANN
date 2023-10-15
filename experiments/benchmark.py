import numpy as np

from experiments.neighbors_utils import *
import logging

# Set var for experiments:
datasets = ['GLOVE']
distances = ['manhattan', 'euclidean', 'chebyshev']
methods = ['FLANN', 'PYNN', 'MASK']
baseline = 'KDTree'
knn = [5, 10, 15]
tg = 1000
nc = 500
r = 250
kmeans_implementation = 'kclust'


def benchmark_recall():

    for da in datasets:

        # Set loging info
        logging.basicConfig(filename='./experiments/logs/' + da + '/' + da + '_knn_recall.log',
                            filemode='w', format='%(asctime)s - %(name)s - %(message)s', level=logging.INFO)
        logging.info('------------------------------------------------------------------------')
        logging.info('                       KNN over %s Dataset RECALL', da)
        logging.info('------------------------------------------------------------------------')
        logging.info('Search of k neighbors over a choosen dataset, using different methods, benchmark.py')
        logging.info('------------------------------------------------------------------------\n')

        logging.info('Distances: %s ', distances)
        logging.info('Methods: %s', methods)
        logging.info('KNN: %s', knn)
        logging.info('MASK params: tg=%s, nc=%s, r=%s, kmeans_implementation=%s\n', tg, nc, r, kmeans_implementation)

        # From a chosen dataset, calculate recalls, store them and print graph
        da_recalls = []
        for di in distances:
            logging.info('------------  %s distance  --------------------\n', di)
            di_recall = []
            for m in methods:
                m_recall = []

                logging.info('-- %s method --\n', m)

                for k in knn:

                    file_name_le = "./experiments/NearestNeighbors/" + da + "/knn_" + da + "_" + str(
                        k) + "_" + di + "_" + baseline + ".hdf5"

                    if m == 'MASK':
                        file_name = "./experiments/NearestNeighbors/" + da + "/knn_" + da + "_" + str(
                            k) + "_" + di + "_" + m + "_tg" + str(tg) + "_nc" + str(nc) + "_r" + str(r) + "_" + str(kmeans_implementation) + ".hdf5"
                    else:
                        file_name = "./experiments/NearestNeighbors/" + da + "/knn_" + da + "_" + str(
                            k) + "_" + di + "_" + m + ".hdf5"

                    ##Arreglo mientras MASK no tenga implementacion con distancias no euclideas

                    #if ((m == 'MASK') & (di != 'euclidean')) | ((m == 'FLANN') & (di == 'chebyshev')):
                    if ((m == 'FLANN') & (di == 'chebyshev')):
                        re = np.nan
                    else:
                        re = recall(da, di, m, k, False, file_name_le, file_name)

                    m_recall.append(re)

                di_recall.append(m_recall)
            da_recalls.append(di_recall)

        # Show results on a graph
        #print_recall_graph(da, distances, methods, knn, da_recalls)
        print_recall_heatmap(da, distances, methods, knn, da_recalls)
        print(da_recalls)


def benchmark_error_rate():

    for da in datasets:

        # Set loging info
        logging.basicConfig(filename='./experiments/logs/' + da + '/' + da + '_knn_error_rate.log',
                            filemode='w', format='%(asctime)s - %(name)s - %(message)s', level=logging.INFO)
        logging.info('------------------------------------------------------------------------')
        logging.info('                       KNN over %s Dataset ERROR RATE', da)
        logging.info('------------------------------------------------------------------------')
        logging.info('Search of k neighbors over a choosen dataset, using different methods, benchmark.py')
        logging.info('------------------------------------------------------------------------\n')

        logging.info('Distances: %s ', distances)
        logging.info('Methods: %s', methods)
        logging.info('KNN: %s', knn)
        logging.info('MASK params: tg=%s, nc=%s, r=%s, kmeans_implementation=%s\n', tg, nc, r, kmeans_implementation)

        # From a chosen dataset, calculate recalls, store them and print graph
        da_error_rate = []
        for di in distances:
            logging.info('------------  %s distance  --------------------', di)
            di_error_rate = []
            for m in methods:
                m_error_rate = []

                logging.info('')
                logging.info('-- %s method --', m)
                logging.info('')

                for k in knn:

                    file_name_le = "./experiments/NearestNeighbors/" + da + "/knn_" + da + "_" + str(k) + "_" + di + "_" + baseline + ".hdf5"

                    if m == 'MASK':
                        file_name = "./experiments/NearestNeighbors/" + da + "/knn_" + da + "_" + str(k) + "_" + di + "_" + m + "_tg" + str(tg) + "_nc" + str(nc) + "_r" + str(r) + "_" + str(kmeans_implementation) + ".hdf5"
                    else:
                        file_name = "./experiments/NearestNeighbors/" + da + "/knn_" + da + "_" + str(k) + "_" + di + "_" + m + ".hdf5"

                    er = error_rate(da, di, m, k, False, file_name_le, file_name)

                    m_error_rate.append(er)

                di_error_rate.append(m_error_rate)
            da_error_rate.append(di_error_rate)


'''

DEPRECATED

def benchmark_execution_time():

    # For each dataset, calculate execution time averages, store them and print graph
    for da in datasets:
        da_ex_times = []
        train_set, test_set = load_train_test_set.load_train_test_h5py(da)
        for di in distances:
            di_ex_times = []
            for m in methods:
                m_ex_times = []
                for k in knn:
                    et = execution_time(da, di, m, k, 1, train_set, test_set)
                    m_ex_times.append(et)

                di_ex_times.append(m_ex_times)
            da_ex_times.append(di_ex_times)

        # Show results on a graph
        print_execution_time(da, distances, methods, knn, da_ex_times)

'''
'''
indices_le, coords_le, dists_le = get_neighbors('gaussian', 'euclidean', 'BruteForce', 5)
indices_mc, coords_mc, dists_mc = get_neighbors('gaussian', 'euclidean', 'MASK', 5)

print(coords_le)
print("cambio-------------------------------------")
print(coords_mc)
'''

benchmark_recall()
