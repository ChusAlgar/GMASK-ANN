from load_train_test_set import *
from neighbors_utils import *
from FLANN_npdist_modulado import FLANN_tree, FLANN_nn


# Set constants for experiments
ncentroids = 128  # At MASK, ncentroids = tam_grupo*n_centroides = 8*16 = 128
knn = 5  # Number of Nearest Neighbors to be found
algorithm = 'kmeans'  # Possible values: linear, kdtree, kmeans, composite, autotuned - default: kdtree
distances = ['euclidean', 'manhattan']   # Possible values euclidean, manhattan, minkowski, max dist (L infinity), hik (histogram intersection kernel), hellinger, cs (chi-square) and kl (Kullback- Leibler)


####### Brute Force KNN with sintetic dataset (Gaussian Clouds)  ############
# Generate dataset and generate train and test sets
train_set, test_set = load_train_test_gaussian()
dataset_name = "gaussian"
'''


####### Brute Force KNN with Geographical Dataset (Municipios) ########
# Load dataset and generate train and test sets
train_set, test_set = load_train_test_municipios()
dataset_name = "municipios"



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
    #print(indices)

    # Store indices, coords and dist into a tridimensional matrix of size vector.size() x 3 x knn
    #knn = zip(indices, coords, dists)

    # Store indices, coords and dist into a hdf5 file
    save_neighbors(indices, coords, dists, knn, "FLANN", dataset_name, d)

    # Print
    #print_knn(train_set, test_set, coords, knn, "FLANN", dataset_name, d)
