import logging
from sklearn.metrics import pairwise_distances
import numpy as np
from scipy.stats import variation, skew, kurtosis
from scipy.spatial import distance_matrix, distance
import other_algorithms.load_train_test_set as lts


# Set constants for experiments
dataset = 'municipios'


# Set log configuration
logging.basicConfig(filename="./logs/analisis_descriptivo_" + dataset + ".log", filemode='w', format='%(asctime)s - %(name)s - %(message)s', level=logging.INFO)
logging.info('------------------------------------------------------------------------')
logging.info('            ' + dataset + " Dataset Descriptive Analysis ")
logging.info('------------------------------------------------------------------------\n')
logging.info("")


# Regarding the dataset name, set the file name to load the train and test set
file_name = "./data/" + str(dataset) + "_train_test_set.hdf5"


# Read data
vector_training, vector_testing = lts.load_train_test_h5py(file_name)
# vector_training, vector_testing = lts.load_train_test(str(dataset))

# Get size and dimensionality of the dataset
logging.info("Size: " + str(vector_training.shape[0]))
logging.info("")

n_dimensiones = vector_training.shape[1]
logging.info("Dimensionality: " + str(n_dimensiones))
logging.info("")

# Build the lists containing values for each dimension of the dataset
da_minmax = []
da_range = []
da_medias = []
da_medianas = []
da_std = []
da_cv = []
da_kur = []
da_asimetrias = []
da_dist = []

# Explore every dimension of the dataset to get relevant statistics
for i in range(0, vector_training.shape[1]):

    dimension = vector_training[:, i]
    print("Dimensión: " + str(i))

    '''
    # Dibujar histograma (resultan una distribucion normal)
    if i == 12:
        plot= plt.hist(dimension, bins='auto')
        plt.show()
    '''

    # Get min and max value
    minmax = (min(dimension), max(dimension))
    da_minmax.append(minmax)

    # Get range of the values
    range = (minmax[1]-minmax[0])
    da_range.append(range)

    # Get mean
    media = np.mean(dimension, axis=0)
    da_medias.append(media)

    # Get median
    mediana = np.median(dimension, axis=0)
    da_medianas.append(mediana)

    # Get standard deviation
    std = np.std(dimension, axis=0)
    da_std.append(std)

    # Coefficient of variation doesn't report relevant information cause, as mean is near 0 (0,1),
    # cv would tend to infinite, so we won't use it
    # cv = variation(dimension, ddof=0)  # Also calculated as cv=std/media
    # da_cv.append(cv)

    # Get kurtosis
    kur = kurtosis(dimension)
    da_kur.append(kur)

    # Get skewness
    asimetria = skew(dimension)
    da_asimetrias.append(asimetria)

    '''
    # Get distance between the point values in an specific dimension 
    # Calculation using sklearn
    dist = pairwise_distances(dimension.reshape(-1, 1))
    mean_dist = np.sum(dist)/dist.size
    da_dist.append(mean_dist)
    '''

    # Get distance between the point values in an specific dimension
    # Calculation using scipy
    # dist = distance_matrix(dimension, np.transpose(dimension), p=2)
    dist = distance.pdist(dimension.reshape(-1, 1), metric='euclidean')
    mean_dist = np.sum(dist)/dist.size
    da_dist.append(mean_dist)

logging.info("------- Descriptive analysis for each dimension of the dataset-------")
logging.info("")
logging.info("MinMax: " + str(da_minmax))
logging.info("")
logging.info("Range: " + str(da_range))
logging.info("")
logging.info("Mean value: " + str(da_medias))
logging.info("")
logging.info("Median value: " + str(da_medianas))
logging.info("")
logging.info("Standard Deviation: " + str(da_std))
logging.info("")
logging.info("Kurtosis: " + str(da_kur))
logging.info("")
logging.info("Skewness (Asimetria): " + str(da_asimetrias))
logging.info("")
logging.info("Mean distance between points (1-d): " + str(da_dist))
logging.info("")


# Distance matrix

# Distance Matrix - distance between every point in the dataset. Calculation using scipy
distances = distance.cdist(vector_training, vector_training, metric='euclidean')

# Min and max distance between points (calculated over a flattened version of the distances matrix)
minmax = (np.min(distances), np.max(distances))

# Mean distance between points
mean_dist = np.sum(distances)/distances.size

# Quantiles of distances between points
q1 = np.quantile(distances, 0.25)
q2 = np.quantile(distances, 0.5)
q3 = np.quantile(distances, 0.75)

logging.info("")
logging.info("------------ Distance Matrix-----------")
logging.info("")
logging.info("MinMax distance: " + str(minmax))
logging.info("")
logging.info("Mean distance between points (all-d): " + str(mean_dist))
logging.info("")
logging.info("Quantiles:  q1=" + str(q1) + "  -  q2=" + str(q2) + "  -  q3=" + str(q3))
logging.info("")
