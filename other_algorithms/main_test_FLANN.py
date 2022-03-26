import numpy as np
import data_test as dt
import pandas as pd
from FLANN_npdist_modulado import FLANN_tree


# Set constants for experiments
nclouds = 8
npc = 200
ncentroids = 128  # En MASK, ncentroids = tam_grupo*n_centroides = 8*16 = 128
normaliza = False
algorithm = 'kmeans'  # Possible values:
distance_type = 'euclidean'  # Possible values: euclidean, manhattan, minkowski, max_dist, hik, hellinger, cs, kl.

# Test FLANN with Image Dataset (MNist)

# Read the csv file and store it into a Pandas DataFrame
data = pd.read_csv('../data/mnist_test.csv', delimiter=',', nrows=None)
images = pd.DataFrame(data)

# Convert the values contained to float
for i in range(len(images)):
    images.iloc[i] = images.iloc[i].astype(np.float)
# print(images.dtypes)

# Remove the label column and convert the DataFrame into a NumpyArray
images = images.drop(columns='label').to_numpy()

# Using FLANN, build the index tree and generate the ncentroids describing the data
centroids = FLANN_tree(images, ncentroids, normaliza, algorithm, distance_type)

'''

# Test FLANN with sintetic dataset (Gaussian Clouds)

# Generate n gaussian clouds and store them into an array
vector, coordx, coordy, puntos_nube = dt.generate_data_gaussian_clouds(nclouds, npc, overlap=True)
vector = np.array(zip(coordx[0], coordy[0]))  # Dont use indexes to build vector!

# Using FLANN, build the index tree and generate the ncentroids describing the data
centroids = FLANN_tree(vector, ncentroids, normaliza, algorithm, distance_type)

# Draw the original gaussian clouds and the centroids computed
dt.pinta(coordx, coordy, np.array([centroids]), npc, nclouds)

'''