import numpy as np
import data_test as dt
from numpy.random import *
import random
import pandas as pd
from sklearn import preprocessing
from FLANN_npdist_modulado import FLANN_tree, FLANN_nn


# Set constants for experiments
nclouds = 8
npc = 200
ncentroids = 128  # At MASK, ncentroids = tam_grupo*n_centroides = 8*16 = 128
normaliza = False
algorithm = 'kmeans'  # Possible values: linear, kdtree, kmeans, composite, saved, autotuned
distance_type = 'euclidean'  # Possible values: euclidean, manhattan, minkowski, max_dist, hik, hellinger, cs, kl.
knn = 2  # Number of Nearest Neighbourgs to be found


####### Test FLANN with Image Dataset (MNist) ########

# DATASET LOADING AND PREPROCESSING

# Read the csv file and store it into a Pandas DataFrame
data = pd.read_csv('../data/mnist_test.csv', delimiter=',', nrows=None)
images = pd.DataFrame(data)

# If normaliza, normalize the dataset
if normaliza:
    images = preprocessing.normalize(images, axis=0, norm='l2')

# Convert the values contained to float
for i in range(len(images)):
    images.iloc[i] = images.iloc[i].astype(np.float)
# print(images.dtypes)

# Remove the label column and convert the DataFrame into a NumpyArray
images = images.drop(columns='label').to_numpy()

# For this experiment, compose the dataset and the query
#seq_buscada = images[0:30]
#images = images[20:]

# Generate a testset of n elements contained in the original dataset
seq_buscada = np.array(random.sample(images.tolist(), 1000))

# print(seq_buscada)

# CENTROIDS AND KNN

# Using FLANN, build the index tree and generate the num_centroids describing the data
centroids = FLANN_tree(images, ncentroids, normaliza, distance_type)

# Using FLANN, find the knn nearest neighbors and provide number of hit/miss
nn = FLANN_nn(images, seq_buscada, knn, distance_type, algorithm)

'''

####### Test FLANN with sintetic dataset (Gaussian Clouds)  ############

# Generate n gaussian clouds and store them into an array
vector, coordx, coordy, puntos_nube = dt.generate_data_gaussian_clouds(nclouds, npc, overlap=True)
vector = np.array(zip(coordx[0], coordy[0]))  # Dont use indexes to build vector!

# If normaliza, normalize the dataset
if normaliza:
    vector = preprocessing.normalize(vector, axis=0, norm='l2')
    
# Using FLANN, build the index tree and generate the ncentroids describing the data
centroids = FLANN_tree(vector, ncentroids, normaliza, algorithm, distance_type)

# Draw the original gaussian clouds and the centroids computed
dt.pinta(coordx, coordy, np.array([centroids]), npc, nclouds)

# Generate a testset of n elements contained in the original gaussian clouds
seq_buscada = np.array(random.sample(vector.tolist(), 1000))

# Using FLANN, find the knn nearest neighbors and provide number of hit/miss
nn = FLANN_nn(images, seq_buscada, knn, algorithm)

'''