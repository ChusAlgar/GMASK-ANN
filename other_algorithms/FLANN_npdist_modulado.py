import numpy
from pyflann import *
from numpy import *
from numpy.random import *
from sklearn import preprocessing
import data_test as dt
import random


def FLANN_tree(vector, ncentroids, normaliza, algorithm, distance_type):

    if normaliza:
        vector = preprocessing.normalize(vector, axis=0, norm='l2')


    # Sets the distance type used. Possible values: euclidean, manhattan, minkowski, max_dist, hik, hellinger, cs, kl.
    set_distance_type(distance_type, order=0)

    # Create a FLANN instance and build and index
    flann = FLANN()
    params = flann.build_index(vector, target_precision=0.9, log_level="info")
    #print params

    # Using kmeans, compute the ncentroids describing the data
    centroids = flann.kmeans(vector, num_clusters=ncentroids, max_iterations=None, mdtype=None)
    #print centroids

    # Generate a testset of n elements contained in the original gaussian clouds
    seq_buscada = numpy.array(random.sample(vector.tolist(), 1000))

    # Find the k (1) nn of each point in the testset using the index previously built
    result, dists = flann.nn(vector, seq_buscada, 1, algorithm=algorithm, branching=32, iterations=7, checks=16)
    #print result
    print dists

    # Count number of hit and ko
    hit = 0.0
    ko = 0.0

    for i in range(dists.size):
        if dists[i] == 0:
            hit = hit+1.0
        else:
            ko = ko+1.0

    # Show percentage of ok/ko on screen
    print("Porcentaje de aciertos: ", hit/dists.size * 100)
    print("Porcentaje de fallos: ", ko/dists.size * 100)

    return centroids


