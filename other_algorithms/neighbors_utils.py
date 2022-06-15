import os
import h5py
import numpy as np
import matplotlib.pyplot as plt


# Store neighbors (indices, coords and dist) into a hdf5 file
def save_neighbors(indices, coords, dists, knn, method, dataset_name, d):

    # Regarding the knn, method, dataset_name and distance choosen, set the file name to store the neighbors
    file_name = "../other_algorithms/NearestNeighbors/" + str(knn) + "nn_" + method + "_" + str(dataset_name) + "_" + str(d) + ".hdf5"

    # Store the 3 different matrix on a hdf5 file
    with h5py.File(file_name, 'w') as f:
        dset1 = f.create_dataset('indices', data=indices)
        dset2 = f.create_dataset('coords', data=coords)
        dset3 = f.create_dataset('dists', data=dists)
        print("Neighbors stored at " + file_name)


# Load neighbors (indices, coords and dist) from a hdf5 file
def get_neighbors(knn, method, dataset_name, d):

    # Regarding the knn, method, dataset_name and distance choosed, set the file name to store the neighbors
    file_name = "../other_algorithms/NearestNeighbors/" + str(knn) + "nn_" + method + "_" + str(
        dataset_name) + "_" + str(d) + ".hdf5"

    # Load indices, coords and dists as 3 independent matrix from the choosn file
    if not os.path.exists(file_name):
        print("File " + file_name + " does not exist")
        return 0, 0, 0
    else:
        with h5py.File(file_name, 'r') as hdf5_file:
            print("Loading neighbors from " + file_name)
            return np.array(hdf5_file['indices']), np.array(hdf5_file['coords']), np.array(hdf5_file['dists'])


# Print train set, test set and neighbors on a file
def print_knn(train_set, test_set, neighbors, knn, method, dataset_name, d):

    # Plot with points, centroids and title
    fig, ax = plt.subplots()
    title = str(knn) + "nn " + method + " " + str(dataset_name) + " " + str(d)
    file_name = "../other_algorithms/NearestNeighbors/Graphics/" + str(knn) + "nn_" + method + "_" + str(dataset_name) + "_" + str(d) + ".eps"
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
def print_knn_fromfile(train_set, test_set, knn, method, dataset_name, d):

    # Load neighbors from a hdf5 file
    indices, coords, dists = get_neighbors(knn, method, dataset_name, d)

    # Plot with points, centroids and title
    return print_knn(train_set, test_set, coords, knn, method, dataset_name, d)

# Recall Benchmark
def recall(knn, method, dataset_name, d):

    # Load neighbors obtained through linear exploration
    indices_le, coords_le, dists_le = get_neighbors(knn, "Brute_Force", dataset_name, d)

    # Load neighbors obtained through the method choosen
    indices_mc, coords_mc, dists_mc = get_neighbors(knn, method, dataset_name, d)

    hit = 0.0
    for i in range(indices_mc.shape[0]):
        hit = hit + len(np.intersect1d(indices_mc[i], indices_le[i]))

    # Recall: %  hit returned vs number of points
    # Show percentage of hit/miss on screen
    print ("---- Case " + str(knn) + " nn applying " + method + " over " + str(dataset_name) + " dataset using " + str(d) + " distance. ----")
    print("Correct neighbors rate: " + str(hit) + "/" + str(float(indices_mc.size)))
    print("Hit percentage: " + str(hit/indices_mc.size * 100) + "%\n\n")




