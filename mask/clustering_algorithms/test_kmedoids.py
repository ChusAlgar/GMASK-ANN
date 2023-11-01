"""
test_kmedoids.py

This is a simple script to test the functionality provided in different packages
implementing the k-medoids algorithm.

https://en.wikipedia.org/wiki/K-medoids

k-medoids is a generalization of k-means, suitable for arbitrary distance functions.
In contrast to the k-means algorithm, k-medoids chooses actual data points as centers (medoids or exemplars).
Because k-medoids minimizes a sum of pairwise dissimilarities instead of a sum of squared Euclidean distances,
it is more robust to noise and outliers than k-means.

The k-medoids problem is NP-hard. There are different algorithms to solve this problem approximately. One of those
algorithms is Partition Around Medoids (PAM), which requires the entire matrix of pairwise distances between all
elements as an input parameter. Therefore, one must be careful about the size of the dataset, since RAM memory
complexity is in the order of O(N^2), where N is the number of elements in the dataset.

Possible implementations to test:
    - The *scikit-learn-extra* package (**Current choice**).
        - Installation: conda install -c conda-forge scikit-learn-extra
        - It implements a KMedoids method with sklearn-compatible API.
        - Included algorithms: alternating (similar to k-means); PAM.

    - The *kmedoids* package (**Future choice**).
        - Installation: conda install -c conda-forge kmedoids
        - It implements several fast methods to solve the k_medoids problem. It also provides a sklearn-like API.
        - Included algorithms:
            - FasterPAM.
            - PAM.
            - Alternating.
            - Etc.

"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

# Data download and preparation
print("Testing KMedoids clustering")
print("Loading dataset...")
mnist_digits = load_digits()
scaled_digits = scale(mnist_digits.data)
n_digits = len(np.unique(mnist_digits.target))

print("Calculating PCA...")
reduced_digits = PCA(n_components=2).fit_transform(scaled_digits)

# Set up selected models
selected_models = [
    # k-medoids, alternate algorithm, L1 distance
    (KMedoids(metric="manhattan", n_clusters=n_digits), "k-medoids (L1)"),

    # k-medoids, alternate algorithm, L2 distance
    (KMedoids(metric="euclidean", n_clusters=n_digits), "k-medoids (L2)"),

    # k-medoids, alternate algorithm, Linf distance (Chebyshev)
    (KMedoids(metric="chebyshev", n_clusters=n_digits), "k-medoids (Linf)")
]

print("Calculating clusters with different distances...")
# Fit each model to data and retrieve clusters medoids
for i, (model, description) in enumerate(selected_models):
    model.fit(reduced_digits)
    medoids = model.cluster_centers_
    medoids_indices = model.medoid_indices_

    print("Results for ", description)
    print("Medoids coordinates:")
    print(medoids)
    print("Medoids indexes:")
    print(medoids_indices)
    print("----------")

print("END")
