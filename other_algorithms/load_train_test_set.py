import numpy as np
import data_test as dt
from sklearn import preprocessing
from data_geo import *
np.set_printoptions(suppress=True)

# Set constants for experiments
nclouds = 8
npc = 100000

normaliza = False
test_set_size = 100


# Generate Gaussian Clouds dataset and generate train and test sets
def load_train_test_gaussian():

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
    index_complete = np.linspace(0, len(gaussian_clouds)-1, len(gaussian_clouds), dtype=int)
    index_training = np.setdiff1d(index_complete, index_testing)
    train_set = gaussian_clouds[index_training]

    return train_set, test_set


# Load Geographical Dataset (Municipios) dataset and generate train and test sets
def load_train_test_municipios():

    # Read the complete dataset from a csv file and store it into a NumpyArray
    data = pd.read_csv('../data/MUNICIPIOS-utf8.csv', sep=';')
    municipios = processDataGeo(data).to_numpy()

    # If normaliza, normalize the dataset
    if normaliza:
        municipios = preprocessing.normalize(municipios, axis=0, norm='l2')

    # For this experiment, compose the test_set (100 elements) and the train_set
    np.random.seed(1234)
    index_testing = np.random.choice(len(municipios), test_set_size, replace=False)
    test_set = municipios[index_testing]
    index_complete = np.linspace(0, len(municipios)-1, len(municipios), dtype=int)
    index_training = np.setdiff1d(index_complete, index_testing)
    train_set = municipios[index_training]

    # Save train_set on a txt
    #a_file = open("test.txt", "w")
    #for row in train_set:
    #    a_file.write((str(row[0]) + " " + str(row[1]) + "\n"))
    #a_file.close()

    return train_set, test_set

# Load Images Dataset (MNIST) dataset and generate train and test sets
def load_train_test_MNIST():

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
