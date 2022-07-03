import os
import h5py
import numpy as np
import data_test as dt
from sklearn import preprocessing
import pandas as pd
np.set_printoptions(suppress=True)

# Set constants for experiments
nclouds = 8
npc = 100000

normaliza = False
test_set_size = 100

####### Load and store train and test set from a h5py file #########


# Store train and test set into a hdf5 file
def save_train_test_h5py(dataset, train_set, test_set):

    # Regarding the knn, method, dataset_name set the file name to store the train and test set
    file_name = "../data/" + str(dataset) + "_train_test_set.hdf5"

    # Store the 2 different sets on a hdf5 file
    with h5py.File(file_name, 'w') as f:
        dset1 = f.create_dataset('train_set', data=train_set)
        dset1 = f.create_dataset('test_set', data=test_set)


# Load train and test set from a hdf5 file
def load_train_test_h5py(dataset):

    # Regarding the knn, method, dataset_name set the file name to store the train and test set
    file_name = "../data/" + str(dataset) + "_train_test_set.hdf5"

    # Load train and test set from the choosen file
    if not os.path.exists(file_name):
        print("File " + file_name + " does not exist")
        return None, None
    else:
        with h5py.File(file_name, 'r') as hdf5_file:
            print("\n ######### Loading train and test set from " + file_name + " #########")
            return np.array(hdf5_file['train_set']), np.array(hdf5_file['test_set'])


####### Generate brand new train and test set #########

def load_train_test(dataset_name):

    print("\n ######### Creating train and test set from " + dataset_name + " dataset #########")

    # Generate Gaussian Clouds dataset and generate train and test sets
    if dataset_name is "gaussian":

        # Generate n gaussian clouds and store them into a NumpyArray
        gaussian_clouds, coordx, coordy, puntos_nube = dt.generate_data_gaussian_clouds(nclouds, npc, overlap=True)
        gaussian_clouds = np.array(gaussian_clouds)

        # If normaliza, normalize the dataset
        if normaliza:
            gaussian_clouds = preprocessing.normalize(gaussian_clouds, axis=0, norm='l2')

        # For this experiment, compose the test_set (100 elements) and the train_set
        np.random.seed(1234)
        index_testing = np.random.choice(len(gaussian_clouds), test_set_size, replace=False)
        test_set = gaussian_clouds[index_testing]
        index_complete = np.linspace(0, len(gaussian_clouds) - 1, len(gaussian_clouds), dtype=int)
        index_training = np.setdiff1d(index_complete, index_testing)
        train_set = gaussian_clouds[index_training]

        return train_set, test_set

    # Load Geographical Dataset (Municipios) dataset and generate train and test sets
    elif dataset_name is "municipios":

        # Read the complete dataset from a csv file and store it into a NumpyArray
        datos = pd.read_csv('../data/MUNICIPIOS-utf8.csv', sep=';')
        municipios = pd.DataFrame(datos, columns=['LONGITUD_ETRS89', 'LATITUD_ETRS89'])
        # index = datos_geo.index
        # cant_ptos = len(index)
        municipios['LONGITUD_ETRS89'] = municipios['LONGITUD_ETRS89'].str.replace(',', '.').astype(np.float)
        municipios['LATITUD_ETRS89'] = municipios['LATITUD_ETRS89'].str.replace(',', '.').astype(np.float)
        municipios = municipios.to_numpy()

        # If normaliza, normalize the dataset
        if normaliza:
            municipios = preprocessing.normalize(municipios, axis=0, norm='l2')

        # For this experiment, compose the test_set (100 elements) and the train_set
        np.random.seed(1234)
        index_testing = np.random.choice(len(municipios), test_set_size, replace=False)
        test_set = municipios[index_testing]
        index_complete = np.linspace(0, len(municipios) - 1, len(municipios), dtype=int)
        index_training = np.setdiff1d(index_complete, index_testing)
        train_set = municipios[index_training]

        # Save train_set on a txt
        # a_file = open("test.txt", "w")
        # for row in train_set:
        #    a_file.write((str(row[0]) + " " + str(row[1]) + "\n"))
        # a_file.close()

        return train_set, test_set

    # Load Images Dataset (MNIST) dataset and generate train and test sets
    elif dataset_name is "MNIST":

        # Read the train_set from a csv file and store it into a Numpy Array
        data = pd.read_csv('../data/mnist_train.csv', delimiter=',', nrows=None)
        train_set = pd.DataFrame(data).drop(columns='label').to_numpy().astype(float)

        # Read the test_set from a csv file and store it into a Numpy Array
        data = pd.read_csv('../data/mnist_test.csv', delimiter=',', nrows=None)
        test_set = pd.DataFrame(data).drop(columns='label').to_numpy().astype(float)

        # For this experiment, compose a reduced test_set of 100 elements
        np.random.seed(1234)
        index_testing = np.random.choice(len(test_set), test_set_size, replace=False)
        test_set = test_set[index_testing]

        # If normaliza, normalize the datasets
        if normaliza:
            train_set = preprocessing.normalize(train_set, axis=0, norm='l2')
            test_set = preprocessing.normalize(test_set, axis=0, norm='l2')

        return train_set, test_set

    else:

        print("Dataset not found")
        return None, None

