import logging
import numpy as np
from scipy.stats import skew, kurtosis
from scipy.spatial import distance
import experiments.load_train_test_set as lts


# Set constants for experiments
dataset = 'wdbc'


# Set log configuration
logging.basicConfig(filename="./experiments/logs/" + dataset + "/analisis_descriptivo_" + dataset + ".log", filemode='w', format='%(asctime)s - %(name)s - %(message)s', level=logging.INFO)
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
    # As it's a 1-d analysis, distance choosen has no impact. We use euclidean for simplicity
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
distances_manhattan = np.array(distance.pdist(vector_training, metric='cityblock'))
distances_euclidean = np.array(distance.pdist(vector_training, metric='euclidean'))
distances_chebyshev = np.array(distance.pdist(vector_training, metric='chebyshev'))

# Min and max distance between points (calculated over a flattened version of the distances matrix)
minmax_manhattan = (np.min(distances_manhattan), np.max(distances_manhattan))
minmax_euclidean = (np.min(distances_euclidean), np.max(distances_euclidean))
minmax_chebyshev = (np.min(distances_chebyshev), np.max(distances_chebyshev))

# Mean distance between points
mean_dist_manhattan = np.sum(distances_manhattan)/distances_manhattan.size
mean_dist_euclidean = np.sum(distances_euclidean)/distances_euclidean.size
mean_dist_chebyshev = np.sum(distances_chebyshev)/distances_chebyshev.size

# Quantiles of distances between points
q1_manhattan = np.quantile(distances_manhattan, 0.25)
q2_manhattan = np.quantile(distances_manhattan, 0.5)
q3_manhattan = np.quantile(distances_manhattan, 0.75)

q1_euclidean = np.quantile(distances_euclidean, 0.25)
q2_euclidean = np.quantile(distances_euclidean, 0.5)
q3_euclidean = np.quantile(distances_euclidean, 0.75)

q1_chebyshev= np.quantile(distances_chebyshev, 0.25)
q2_chebyshev = np.quantile(distances_chebyshev, 0.5)
q3_chebyshev = np.quantile(distances_chebyshev, 0.75)

logging.info("")
logging.info("------------ Distance Matrixes (built using several distance metrics)-----------")
logging.info("")
logging.info("-------- Manhattan distance --------")
logging.info("")
logging.info("MinMax distance: " + str(minmax_manhattan))
logging.info("")
logging.info("Mean distance between points (all-d): " + str(mean_dist_manhattan))
logging.info("")
logging.info("Quantiles:  q1=" + str(q1_manhattan) + "  -  q2=" + str(q2_manhattan) + "  -  q3=" + str(q3_manhattan))

logging.info("")
logging.info("-------- Euclidean distance --------")
logging.info("")
logging.info("MinMax distance: " + str(minmax_euclidean))
logging.info("")
logging.info("Mean distance between points (all-d): " + str(mean_dist_euclidean))
logging.info("")
logging.info("Quantiles:  q1=" + str(q1_euclidean) + "  -  q2=" + str(q2_euclidean) + "  -  q3=" + str(q3_euclidean))

logging.info("")
logging.info("-------- Chebyshev distance --------")
logging.info("")
logging.info("MinMax distance: " + str(minmax_chebyshev))
logging.info("")
logging.info("Mean distance between points (all-d): " + str(mean_dist_chebyshev))
logging.info("")
logging.info("Quantiles:  q1=" + str(q1_chebyshev) + "  -  q2=" + str(q2_chebyshev) + "  -  q3=" + str(q3_chebyshev))
logging.info("")
