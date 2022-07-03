import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import csv


# Store coordinates on a csv file
def save_coordinates_csv(filename, coords):
    with open(str(filename) + ".csv", 'w') as file:
        writer = csv.writer(file)
        writer.writerows(coords)


# Store neighbors (indices, coords and dist) into a hdf5 file
def save_neighbors(indices, coords, dists, dataset_name, d, method, knn):

    # Regarding the knn, method, dataset_name and distance choosen, set the file name to store the neighbors
    file_name = "../other_algorithms/NearestNeighbors/" + method + "/" + str(dataset_name) + "_" + str(d) + "_" + method + "_" + str(knn) + "nn.hdf5"

    # Store the 3 different matrix on a hdf5 file
    with h5py.File(file_name, 'w') as f:
        dset1 = f.create_dataset('indices', data=indices)
        dset2 = f.create_dataset('coords', data=coords)
        dset3 = f.create_dataset('dists', data=dists)
        print("Neighbors stored at " + file_name)


# Load neighbors (indices, coords and dist) from a hdf5 file
def load_neighbors(dataset_name, d, method, knn, tg, nc):

    # Regarding the knn, method, dataset_name and distance choosen, set the file name to store the neighbors
    if method is 'MASK':
        file_name = "../other_algorithms/NearestNeighbors/" + method + "/tg" + str(tg) + "nc" + str(nc) + "/" + str(dataset_name) \
                    + "_" + str(d) + "_" + method + "_" + str(knn) + "nn.hdf5"
    else:
        file_name = "../other_algorithms/NearestNeighbors/" + method + "/" + str(dataset_name) + "_" + str(d) + "_" + method + "_" + str(knn) + "nn.hdf5"

    # Load indices, coords and dists as 3 independent matrix from the choosen file
    if not os.path.exists(file_name):
        print("File " + file_name + " does not exist")
        return None, None, None
    else:
        with h5py.File(file_name, 'r') as hdf5_file:
            print("Loading neighbors from " + file_name)
            return np.array(hdf5_file['indices']), np.array(hdf5_file['coords']), np.array(hdf5_file['dists'])


# Print train set, test set and neighbors on a file
def print_knn(train_set, test_set, neighbors, dataset_name, d, method, knn):

    # Plot with points, centroids and title
    fig, ax = plt.subplots()
    title = str(dataset_name) + "_" + str(d) + "_" + method + "_" + str(knn) + "nn"
    file_name = "../other_algorithms/NearestNeighbors/Graphics/" + str(dataset_name) + "_" + str(d) + "_" + method + "_" + str(knn) + "nn.eps"
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


# Print train set, test set and neighbors loaded into a file
def print_knn_fromfile(train_set, test_set, dataset_name, d, method, knn, tg, nc):

    # Load neighbors from a hdf5 file
    indices, coords, dists = load_neighbors(dataset_name, d, method, knn, tg, nc)

    # Plot with points, centroids and title
    return print_knn(train_set, test_set, coords, dataset_name, d, method, knn)


# Recall Benchmark
def recall(dataset_name, d, method, knn, tg, nc):

    # Load neighbors obtained through linear exploration
    indices_le, coords_le, dists_le = load_neighbors(dataset_name, d, "Brute_Force", knn, tg, nc)

    # Load neighbors obtained through the method choosen
    indices_mc, coords_mc, dists_mc = load_neighbors(dataset_name, d, method, knn, tg, nc)

    if indices_mc is None:
        return None

    hit = 0.0
    for i in range(indices_mc.shape[0]):
        hit = hit + len(np.intersect1d(indices_mc[i].astype(int), indices_le[i]))

    # Recall: %  hit returned vs number of points
    rec = hit/indices_mc.size * 100

    # Show percentage of hit/miss on screen
    print ("---- Case " + str(knn) + " nn applying " + method + " over " + str(dataset_name) + " dataset using " + str(d) + " distance. ----")
    print("Correct neighbors rate: " + str(hit) + "/" + str(float(indices_mc.size)))
    print("Hit percentage: " + str(rec) + "%\n\n")

    return rec


# Build a graph to show recall results
def print_recall(dataset, distances, methods, knn, recalls):

    fig, axs = plt.subplots(1, 3, figsize=(9, 4), sharey=True)
    fig.subplots_adjust(top=0.75)

    for i in range(len(distances)):
        for j in range(len(methods)):
            axs[i].plot(knn, recalls[i][j], label=methods[j], marker='o')
            axs[i].set_title(distances[i], pad=7)

    fig.legend(methods, loc='center right', title='Method')
    fig.suptitle(dataset + " dataset recall (%)", fontsize=20, y= 0.95)
    plt.ylim([0, 105])
    plt.show()


# Compare intersection percentage between neighbours found by two different methods
def compare(dataset_name, d, method1, method2, knn, tg, nc):

    # Load neighbors obtained through first method
    indices_le, coords_le, dists_le = load_neighbors(dataset_name, d, method1, knn, tg, nc)

    # Load neighbors obtained through the second method choosen
    indices_mc, coords_mc, dists_mc = load_neighbors(dataset_name, d, method2, knn, tg, nc)

    hit = 0.0
    for i in range(indices_mc.shape[0]):
        hit = hit + len(np.intersect1d(indices_mc[i], indices_le[i]))

    # Compare: %  hit returned vs number of points
    ip = hit/indices_mc.size * 100
    # Show percentage of hit/miss on screen
    print ("---- Case " + str(knn) + " nn within " + method1 + " and " + method2 + " over " + str(dataset_name) + " dataset using " + str(d) + " distance. ----")
    print("Same neighbors rate: " + str(hit) + "/" + str(float(indices_mc.size)))
    print("Intersection percentage: " + str(ip) + "%\n\n")

    return ip


