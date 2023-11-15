import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import csv
import logging
import experiments.load_train_test_set as load_train_test_set
import re
import seaborn as sns
import itertools


# Store query points and its neighbors on a csv file
# arguments: file name, train_test hdf5 file, neighbors file
# save_csv("./experiments/municipios_5_euclidean_FLANN", "./data/municipios_train_test_set.hdf5", "./experiments/NearestNeighbors/municipios/knn_municipios_5_euclidean_FLANN.hdf5")
def save_csv(filename, train_test_file, neighbors_file):
    with open(str(filename) + ".csv", 'w') as file:
        writer = csv.writer(file)
        header = ['index', 'query_point', 'neighbors']
        writer.writerow(header)

        train_test, test_set = load_train_test_set.load_train_test_h5py(train_test_file)
        indices_n, coords_n, dists_n = load_neighbors(neighbors_file)


        num_neighbors = re.split('_|\.',  neighbors_file)[3]

        for i in range(0, len(test_set)):
            writer.writerow([i, test_set[i], str(coords_n[i].tolist()).replace(",", "")])


# Store only coordinates on a csv file
def save_coordinates_csv(filename, coords):
    with open(str(filename) + ".csv", 'w') as file:
        writer = csv.writer(file)
        writer.writerows(coords)


# Store neighbors (indices, coords and dist) into a hdf5 file
def save_neighbors(indices, coords, dists, file_name):

    # Store the 3 different matrix on a hdf5 file
    with h5py.File(file_name, 'w') as f:
        dset1 = f.create_dataset('indices', data=indices)
        dset2 = f.create_dataset('coords', data=coords)
        dset3 = f.create_dataset('dists', data=dists)
        print("Neighbors stored at " + file_name)
        logging.info("Neighbors stored at " + file_name)


# Load neighbors (indices, coords and dist) from a hdf5 file
def load_neighbors(file_name):

    # Load indices, coords and dists as 3 independent matrix from the choosen file
    if not os.path.exists(file_name):

        print("File " + file_name + " does not exist")
        logging.info("File " + file_name + " does not exist\n")

        return None, None, None

    else:
        with h5py.File(file_name, 'r') as hdf5_file:

            print("Loading neighbors from " + file_name)
            logging.info("Loading neighbors from " + file_name)

            return np.array(hdf5_file['indices']), np.array(hdf5_file['coords']), np.array(hdf5_file['dists'])


# Print train set, test set and neighbors on a file
def print_knn(train_set, test_set, neighbors, dataset_name, d, method, knn, file_name):

    # Plot with points, centroids and title
    fig, ax = plt.subplots()
    title = str(dataset_name) + "_" + str(d) + "_" + method + "_" + str(knn) + "nn"
    plt.title(title)

    train_set = zip(*train_set)
    test_set = zip(*test_set)

    ax.scatter(train_set[0], train_set[1], marker='o', s=1, color='#1f77b4', alpha=0.5)

    for point in neighbors:
        point = zip(*point)
        ax.scatter(point[0], point[1], marker='o', s=1, color='#949494', alpha=0.5)

    ax.scatter(test_set[0], test_set[1], marker='o', s=1, color='#ff7f0e', alpha=0.5)

    plt.savefig(file_name)
    print("Train set, test set and neighbors printed at " + file_name)

    return plt.show()


# Recall Benchmark
def recall(dataset_name, d, method, k, same_train_test=False, file_name_le=None, file_name=None):

    # Recall in Exhaustive Point Query (query points are the same from training set)
    if same_train_test:

        # Load neighbors obtained through the method choosen
        indices_mc, coords_mc, dists_mc = load_neighbors(file_name)

        '''
        hit = 0.0
        for i in range(indices_mc.shape[0]):
            if indices_mc[i] == i:
                hit = hit + 1
        '''

        # Count number of 1-neighbor which are the same as the point searched
        #hit = map(lambda x, y: x == y, list(indices_mc), range(indices_mc.shape[0])).count(True)


    # Recall in query points different from training set
    else:
        # Load neighbors obtained through linear exploration
        indices_le, coords_le, dists_le = load_neighbors(file_name_le)

        # Load neighbors obtained through the method choosen
        indices_mc, coords_mc, dists_mc = load_neighbors(file_name)

        '''
        hit = 0.0
        for i in range(indices_mc.shape[0]):
            hit = hit + len(np.intersect1d(indices_mc[i].astype(int), indices_le[i]))
        '''

        # Count number of 1-neighbor which are the same as the point searched
        hit = sum(map(lambda x, y: len(np.intersect1d(x.astype(int), y)), list(indices_mc), list(indices_le)))

    # Recall: %  hit returned vs number of points
    rec = hit / indices_mc.size * 100


    # Show percentage of hit/miss on screen and save information on log file
    '''
    print ("---- Case " + str(k) + " nn applying " + method + " over " + str(dataset_name) + " dataset using " + str(d) + " distance. ----")
    print("Correct neighbors rate: " + str(hit) + "/" + str(float(indices_mc.size)))
    print("Hit percentage: " + str(rec) + "%\n\n")
    logging.info("---- Case " + str(k) + " nn applying " + method + " over " + str(dataset_name) + " dataset using " + str(d) + " distance. ----")
    '''
    logging.info("Correct neighbors rate: " + str(hit) + "/" + str(float(indices_mc.size)))
    logging.info("Hit percentage: " + str(rec) + "%\n\n")

    return rec


# Build a graph to show recall results
def print_recall_graph(dataset, distances, methods, k, recalls):

    fig, axs = plt.subplots(1, 3, figsize=(9, 4), sharey=True)
    fig.subplots_adjust(top=0.75)

    for i in range(len(distances)):
        for j in range(len(methods)):
            axs[i].plot(k, recalls[i][j], label=methods[j], marker='o')
            axs[i].set_title(distances[i], pad=7)

    fig.legend(methods, loc='center right', title='Method')
    fig.suptitle(dataset + " dataset - Recall (%)", fontsize=20, y= 0.95)
    plt.ylim([0, 105])
    plt.show()

# Build a heatmap to show recall results
def print_recall_heatmap(dataset, distances, methods, k, recalls):


    re_ma = np.asarray(list(itertools.chain(*recalls[0])))
    re_eu = np.asarray(list(itertools.chain(*recalls[1])))
    re_ch = np.asarray(list(itertools.chain(*recalls[2])))

    # setting the dimensions of the plot
    fig, ax = plt.subplots(figsize=(16, 16))

    # Create a mask to hide null (np.nan) values from heatmap
    mask = [re_ma, re_eu, re_ch]==np.nan

    # Heatmap
    h = sns.heatmap([re_ma, re_eu, re_ch], annot=True, annot_kws={"size": 20}, fmt='.3g', yticklabels=distances, xticklabels=k+k+k, cmap="icefire", mask=mask, vmin=0, vmax=100)

    #Colorbar
    h.collections[0].colorbar.set_label('Recall (%)', labelpad=30, fontsize=25)
    h.collections[0].colorbar.ax.tick_params(labelsize=20)

    #Title
    h.axes.set_title(str("Dataset " + dataset), fontsize=30, pad=35)

    # Axis x and y (knn and distance)
    h.set_xlabel("K-Neighbors", fontsize=25, labelpad=30)
    h.set_ylabel("Distance", fontsize=25, labelpad=40)
    h.tick_params(axis='both', which='major', labelsize=20)

    # Axis twin (method)
    hb = h.twiny()
    hb.set_xticks(range(len(methods)))
    #hb.set(xticklabels=methods)
    hb.set_xticklabels(methods, ha='center')
    hb.set_aspect(aspect=1.3)
    hb.set_xlabel("Method", fontsize=25, labelpad=30)
    hb.tick_params(axis='both', which='major', labelsize=20)
    
    


    # Show heatmap
    plt.show()

    '''
    fig, ax = plt.subplots()
    im = ax.imshow([re_ma, re_eu, re_ch])

    # Show all ticks and label them with the respective list entries
    #ax.set_xticks(np.arange(len(farmers)), labels=farmers)
    ax.set_yticks(np.arange(len(distances)), labels=distances)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")


    # Loop over data dimensions and create text annotations.
    for i in range(len(vegetables)):
        for j in range(len(farmers)):
            text = ax.text(j, i, harvest[i, j],
                           ha="center", va="center", color="w")
    
    ax.set_title("Harvest of local farmers (in tons/year)")
    fig.tight_layout()
    plt.show()
    
    '''

# Error rate
def error_rate(dataset_name, d, method, knn, same_train_test=False, file_name_le=None, file_name=None):

    # Error rate in Exhaustive Point Query when query points are the same from training set
    if same_train_test:

        # Load neighbors obtained through the method choosen
        indices_mc, coords_mc, dists_mc = load_neighbors(file_name)

        '''
        hit = 0.0
        for i in range(indices_mc.shape[0]):
            if indices_mc[i] == i:
                hit = hit + 1
        '''

        # Count number of 1-neighbor which are the same as the point searched
        hit = map(lambda x, y: x == y, list(indices_mc), range(indices_mc.shape[0])).count(True)

    # Error rate in Exhaustive Point Query when query points are the same from training set
    else:
        # Load neighbors obtained through linear exploration
        indices_le, coords_le, dists_le = load_neighbors(file_name_le)

        # Load neighbors obtained through the method choosen
        indices_mc, coords_mc, dists_mc = load_neighbors(file_name)

        '''
        hit = 0.0
        for i in range(indices_mc.shape[0]):
            hit = hit + len(np.intersect1d(indices_mc[i].astype(int), indices_le[i]))
        '''

        # Count number of 1-neighbor which are the same as the point searched
        # We set assume_unique=True at np.intersectid(...) to accelerate the calculation
        # bc the compared lists are always uniques (any element would ever appear twice as neighbor for the same point)
        hit = sum(map(lambda x, y: len(np.intersect1d(x.astype(int), y, assume_unique=True)), list(indices_mc), list(indices_le)))

    # Compare: % miss returned vs number of points
    er = (1 - hit / float(indices_mc.size)) * 100

    # Show percentage of hit/miss on screen an save information on a log file
    '''print("---- Case " + str(knn) + " nn within " + method + " over " + str(dataset_name) + " dataset using " + str(d) + " distance. ----")
    print("Found points rate: " + str(hit) + "/" + str(float(indices_mc.size)))
    print("Error rate: " + str(er) + "%\n\n")
    logging.info("")
    logging.info("---- Case " + str(knn) + " nn within " + method + " over " + str(dataset_name) + " dataset using " + str(d) + " distance. ----")
    '''
    logging.info("Found points rate: " + str(hit) + "/" + str(float(indices_mc.size)))
    logging.info("Error rate: " + str(er) + "%")

    return er



# Compare intersection percentage between neighbors found by two different methods
def compare(dataset_name, d, method1, method2, knn, file_name1=None, file_name2=None):

    # Load neighbors obtained through first method
    indices_le, coords_le, dists_le = load_neighbors(file_name1)

    # Load neighbors obtained through the second method choosen
    indices_mc, coords_mc, dists_mc = load_neighbors(file_name2)

    # Count number of 1-neighbor which are calculated as the same by both methods
    hit = sum(map(lambda x, y: len(np.intersect1d(x.astype(int), y)), list(indices_mc), list(indices_le)))


    # Compare: %  hit returned vs number of points
    ip = hit/indices_mc.size * 100

    # Show percentage of hit/miss on screen an save information on a log file
    print ("---- Case " + str(knn) + " nn within " + method1 + " and " + method2 + " over " + str(dataset_name) + " dataset using " + str(d) + " distance. ----")
    print("Same neighbors rate: " + str(hit) + "/" + str(float(indices_mc.size)))
    print("Intersection percentage: " + str(ip) + "%\n\n")
    logging.info("---- Case " + str(knn) + " nn within " + method1 + " and " + method2 + " over " + str(dataset_name) + " dataset using " + str(d) + " distance. ----")
    logging.info("Same neighbors rate: " + str(hit) + "/" + str(float(indices_mc.size)))
    logging.info("Intersection percentage: " + str(ip) + "%\n\n")

    return ip


'''
DEPRECATED
# Execution time benchmark
def execution_time(dataset_name, distance, method, k, rep, train_set, test_set):

    if method is 'BruteForce':

        t1 = timeit.Timer(lambda: bruteforce.bruteforce_nn_index(train_set, k, distance, 'brute'))
        index_time = t1.timeit(rep)/rep

        knn_index = bruteforce.bruteforce_nn_index(train_set, k, distance, 'brute')
        t2 = timeit.Timer(lambda: bruteforce.bruteforce_nn_search(train_set, test_set, k, distance, 'brute', knn_index))
        search_time = t2.timeit(rep) / rep

    elif method is 'FLANN':

        t1 = timeit.Timer(lambda: flann.FLANN_nn_index(train_set, 128, distance, 'kmeans'))
        index_time = t1.timeit(rep) / rep

        t2 = timeit.Timer(lambda: flann.FLANN_nn_search(train_set, test_set, k, distance, 'kmeans'))
        search_time = t2.timeit(rep) / rep

    elif method is 'PYNN':
        index_time, search_time = None, None

    elif method is 'LSH':
        index_time, search_time = None, None

    elif method is 'MASK':
        index_time, search_time = None, None

    # Show time average on screen:
    print ("---- Case " + str(k) + " nn applying " + method + " over " + str(dataset_name) + " dataset using " + str(distance) + " distance. ----")
    print("Index time av: " + str(index_time) + " / Search time av: " + str(search_time))

    return index_time, search_time
'''

# Build a graph to show execution time results
def print_execution_time(dataset, distances, methods, k, ex_times):

    fig, axs = plt.subplots(2, 3, figsize=(9, 4), sharey=True)
    fig.subplots_adjust(top=0.75)

    for i in range(len(distances)):
        for j in range(len(methods)):
            axs[0][i].set_title(distances[i], pad=7)
            axs[0][0].set_ylabel('Indexation time')
            axs[1][0].set_ylabel('Search time')
            for z in range (2):
                aux_extimes = np.transpose(ex_times[i][j])
                axs[z][i].plot(k, aux_extimes[z], label=methods[j], marker='o')

    fig.legend(methods, loc='center right', title='Method')
    fig.suptitle(dataset + " dataset - Execution time avevarage (s)", fontsize=20, y=0.95)
    plt.show()