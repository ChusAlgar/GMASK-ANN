from experiments.neighbors_utils import save_csv, load_train_test_set
import experiments.neighbors_utils as neighbors_utils
from load_train_test_set import load_train_test

#save_csv("./experiments/MNIST_5_manhattan_MASK", "./data/MNIST_train_test_set.hdf5", "./experiments/NearestNeighbors/MNIST/knn_MNIST_5_manhattan_MASK_tg1000_nc500_r4500_kclust.hdf5")

#save_csv("./experiments/MNIST_5_euclidean_MASK_sk", "./data/MNIST_train_test_set.hdf5", "./experiments/NearestNeighbors/MNIST/knn_MNIST_5_euclidean_MASK_tg1000_nc500_r4500_sklearn.hdf5")
"""
load_train_test('MNIST')
knnKD = neighbors_utils.load_neighbors('experiments/NearestNeighbors/MNIST/knn_MNIST_5_chebyshev_KDTree.hdf5')
knnMASK = neighbors_utils.load_neighbors('experiments/NearestNeighbors/MNIST/knn_MNIST_5_chebyshev_MASK_tg1000_nc500_r80000_kmedoids_fastkmedoids.hdf5')

with open('experiments/test1', 'w') as f:
    print(knnKD[2][0], file=f)
    print(knnMASK[2][0], file=f)
    
"""