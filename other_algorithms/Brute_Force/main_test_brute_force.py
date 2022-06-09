from load_train_test_set import *
from neighbors_utils import *
from brute_force_npdist import brute_force_nn

# Set constants for experiments
ncentroids = 128  # At MASK, ncentroids = tam_grupo*n_centroides = 8*16 = 128
knn = 15  # Number of Nearest Neighbors to be found
algorithm = 'brute'  # Possible values 'auto', 'ball_tree', 'kd_tree', 'brute'
distances = ['euclidean', 'manhattan', 'chebyshev']   # Possible values:
                             # From scikit-learn: ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan']. These metrics support sparse matrix inputs. ['nan_euclidean'] but it does not yet support sparse matrices
                             # From scipy.spatial.distance: ['braycurtis', 'canberra', 'chebyshev', 'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule']bThese metrics do not support sparse matrix inputs.


# Load the train and test sets to carry on the experiment

'''
####### Brute Force KNN with sintetic dataset (Gaussian Clouds)  ############
# Generate dataset and generate train and test sets
train_set, test_set = load_train_test_gaussian()
dataset_name = "gaussian"



####### Brute Force KNN with Geographical Dataset (Municipios) ########
# Load dataset and generate train and test sets
train_set, test_set = load_train_test_municipios()
dataset_name = "municipios"


'''
####### Brute Force KNN with Image Dataset (MNist) ########
# Load dataset and generate train and test sets
train_set, test_set = load_train_test_MNIST()
dataset_name = "MNIST"



# Find the knn from the train_set of the elements contained in the test_set, using distances choosen
for d in distances:

    # Using Brute Force Algorithm, find the k nearest neighbors
    indices, coords, dists = brute_force_nn(train_set, test_set, knn, d, algorithm)

    # Store indices, coords and dist into a tridimensional matrix of size vector.size() x 3 x knn (1600 x 3 x 15)
    #knn = zip(indices, coords, dists)

    # Store indices, coords and dist into a hdf5 file
    save_neighbors(indices, coords, dists, knn, "Brute_Force", dataset_name, d)

    # Print
    #print_knn(train_set, test_set, coords, knn, "Brute_Force", dataset_name, d)

