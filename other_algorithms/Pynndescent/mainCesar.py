import numpy as np
import pynndescent
import timeit
from neighbors_utils import save_neighbors
from load_train_test_set import load_train_test, load_train_test_h5py


def save_case_MNIST(train, neighbors, file_name, k):
    vecinos = np.empty([neighbors[0].shape[0], neighbors[0].shape[1]])
    coordenadas = np.empty([neighbors[0].shape[0], neighbors[0].shape[1], 784])
    distancias = np.empty([neighbors[1].shape[0], neighbors[1].shape[1]])
    for i in range(len(neighbors[0])):
        vecinos[i] = neighbors[0][i]
        coordenadas[i] = train[neighbors[0][i]]
        distancias[i] = neighbors[1][i]

    # Regarding the knn, method, dataset_name and distance choosen, set the file name to store the neighbors
    file_name = "./NearestNeighbors/PYNN/" + str(prueba) + "_" + str(
        distancia) + "_PYNN_" + str(k) + "nn.hdf5"
    save_neighbors(vecinos, coordenadas, distancias, file_name)


def save_case(train, neighbors, file_name, k):
    vecinos = np.empty([neighbors[0].shape[0], neighbors[0].shape[1]])
    coordenadas = np.empty([neighbors[0].shape[0], neighbors[0].shape[1], 2])
    distancias = np.empty([neighbors[1].shape[0], neighbors[1].shape[1]])
    for i in range(len(neighbors[0])):
        vecinos[i] = neighbors[0][i]
        coordenadas[i] = train[neighbors[0][i]]
        distancias[i] = neighbors[1][i]

    # Regarding the knn, method, dataset_name and distance choosen, set the file name to store the neighbors
    file_name = "./NearestNeighbors/PYNN/" + str(prueba) + "_" + str(
        distancia) + "_PYNN_" + str(k) + "nn.hdf5"
    save_neighbors(vecinos, coordenadas, distancias, file_name)


def launch_test(prueba, distancia):

    rep = 10
    file_name=""

    if prueba == "geo":
        # Regarding the dataset_name set the file name to load the train and test set
        file_name = "./data/municipios_train_test_set.hdf5"
    elif prueba == "MNIST":
        # Regarding the dataset_name set the file name to load the train and test set
        file_name = "./data/MNIST_train_test_set.hdf5"
    elif prueba == "Gaussian":
        nclouds = 8
        npc = 100000
        # Regarding the dataset_name set the file name to load the train and test set
        file_name = "./data/gaussian_train_test_set.hdf5"

    # Load the train and test sets of each dataset to carry on the experiment
    train_set, test_set = load_train_test_h5py(file_name)

    if distancia == "Euclidean":
        # Euclidean
        t1 = timeit.Timer(lambda: pynndescent.NNDescent(train_set))
        index_time = t1.timeit(rep) / rep
        index = pynndescent.NNDescent(train_set)

        t1A = timeit.Timer(lambda: index.query(test_set, k=5))
        t1A_time = t1A.timeit(rep) / rep

        t1b = timeit.Timer(lambda: index.query(test_set, k=10))
        t1b_time = t1b.timeit(rep) / rep

        t1C = timeit.Timer(lambda: index.query(test_set, k=15))
        t1C_time = t1C.timeit(rep) / rep

        # Caso 1-A
        neighbors1A = index.query(test_set, k=5)
        save_case(train_set, neighbors1A, "./results/" + prueba + "/caso1A.hdf5", k=5)

        # Caso 1-B
        neighbors1B = index.query(test_set, k=10)
        save_case(train_set, neighbors1B, "./results/" + prueba + "/caso1B.hdf5", k=10)

        # Caso 1-C
        neighbors1C = index.query(test_set, k=15)
        save_case(train_set, neighbors1C, "./results/" + prueba + "/caso1C.hdf5", k=15)

    elif distancia == "Manhattan":
        # Manhattan
        t1_manhattan = timeit.Timer(lambda: pynndescent.NNDescent(train_set, metric="manhattan"))
        index_time_manhattan = t1_manhattan.timeit(rep) / rep

        index_manhattan = pynndescent.NNDescent(train_set, metric="manhattan")

        t2A = timeit.Timer(lambda: index_manhattan.query(test_set, k=5))
        t2A_time = t2A.timeit(rep) / rep

        t2B = timeit.Timer(lambda: index_manhattan.query(test_set, k=10))
        t2B_time = t2B.timeit(rep) / rep

        t2C = timeit.Timer(lambda: index_manhattan.query(test_set, k=15))
        t2C_time = t2C.timeit(rep) / rep

        # Caso 2-A
        neighbors2A = index_manhattan.query(test_set, k=5)
        save_case(train_set, neighbors2A, "./results/" + prueba + "/caso2A.hdf5", k=5)

        # Caso 2-B
        neighbors2B = index_manhattan.query(test_set, k=10)
        save_case(train_set, neighbors2B, "./results/" + prueba + "/caso2B.hdf5", k=10)

        # Caso 2-C
        neighbors2C = index_manhattan.query(test_set, k=15)
        save_case(train_set, neighbors2C, "./results/" + prueba + "/caso2C.hdf5", k=15)

    elif distancia == "Chebyshev":
        # Chebyshev
        t1_chebyshev = timeit.Timer(lambda: pynndescent.NNDescent(train_set, metric="chebyshev"))
        index_time_chebyshev = t1_chebyshev.timeit(rep) / rep

        index_chebyshev = pynndescent.NNDescent(train_set, metric="chebyshev")

        t3A = timeit.Timer(lambda: index_chebyshev.query(test_set, k=5))
        t3A_time = t3A.timeit(rep) / rep

        t3B = timeit.Timer(lambda: index_chebyshev.query(test_set, k=10))
        t3B_time = t3B.timeit(rep) / rep

        t3C = timeit.Timer(lambda: index_chebyshev.query(test_set, k=15))
        t3C_time = t3C.timeit(rep) / rep

        # Caso 3-A
        neighbors3A = index_chebyshev.query(test_set, k=5)
        save_case(train_set, neighbors3A, "./results/" + prueba + "/caso3A.hdf5", k=5)

        # Caso 3-B
        neighbors3B = index_chebyshev.query(test_set, k=10)
        save_case(train_set, neighbors3B, "./results/" + prueba + "/caso3B.hdf5", k=10)

        # Caso 3-C
        neighbors3C = index_chebyshev.query(test_set, k=15)
        save_case(train_set, neighbors3C, "./results/" + prueba + "/caso3C.hdf5", k=15)


# Set constants for experiments
prueba = "Gaussian"  # geo, MNIST, Gaussian
distancia = "Euclidean"  # Euclidean, Manhattan, Chebyshev
normaliza = False
test_set_size = 100
launch_test(prueba, distancia)
