from load_train_test_set import *
from save_get_neighbors import *
from FLANN_npdist_modulado import FLANN_tree, FLANN_nn


# Set constants for experiments
ncentroids = 128  # At MASK, ncentroids = tam_grupo*n_centroides = 8*16 = 128
knn = 15  # Number of Nearest Neighbors to be found
algorithm = 'kmeans'  # Possible values: linear, kdtree, kmeans, composite, autotuned - default: kdtree
distances = ['euclidean']   # Possible values: # Possible values: euclidean, manhattan, minkowski, max_dist, hik, hellinger, cs, kl. - default:


####### Brute Force KNN with sintetic dataset (Gaussian Clouds)  ############
# Generate dataset and generate train and test sets
train_set, test_set = load_train_test_gaussian()
dataset_name = "gaussian"


'''
####### Brute Force KNN with Geographical Dataset (Municipios) ########
# Load dataset and generate train and test sets
train_set, test_set = load_train_test_municipios()
dataset_name = "municipios"
'''

'''
####### Brute Force KNN with Image Dataset (MNist) ########
# Load dataset and generate train and test sets
train_set, test_set = load_train_test_MNIST()
dataset_name = "MNIST"
'''

# Generate index and centroids
#  and find the knn from the train_set of the elements contained in the test_set, using distances choosen
for d in distances:

    # Using FLANN, build the index tree and generate the num_centroids describing the data
    centroids = FLANN_tree(train_set, ncentroids, d, algorithm)

    # Using FLANN, find the knn nearest neighbors
    indices, coords, dists = FLANN_nn(train_set, test_set, knn, d, algorithm)
    print(indices)

    # Store indices, coords and dist into a tridimensional matrix of size vector.size() x 3 x knn
    #knn = zip(indices, coords, dists)

    # Store indices, coords and dist into a hdf5 file
    file_name = "../other_algorithms/NearestNeighbors/" + str(knn) + "nn_FLANN_" + str(dataset_name) + "_" + str(d)
    save_neighbors(indices, coords, dists, file_name)

    # Draw the original gaussian clouds and the centroids computed
    #dt.pinta(coordx, coordy, np.array([centroids]), npc, nclouds)