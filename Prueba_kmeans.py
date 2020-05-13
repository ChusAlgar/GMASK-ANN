from sklearn.cluster import KMeans
import numpy as np

'''
X = np.array([[1, 2, 3, 4], [1, 4, 5, 6], [1, 0, 5, 5], [4, 2, 1 ,1], [4, 4, 0, 0], [4, 0, 2, 1]])
print(X.ndim)
print(X.shape)
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
print('labels: ', kmeans.labels_)
#print(kmeans.predict([[0, 0], [4, 4]]))
print(kmeans.cluster_centers_)'''

from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

Data = {
    'x': [25, 34, 22, 27, 33, 33, 31, 22, 35, 34, 67, 54, 57, 43, 50, 57, 59, 52, 65, 47, 49, 48, 35, 33, 44, 45, 38,
          43, 51, 46],
    'y': [79, 51, 53, 78, 59, 74, 73, 57, 69, 75, 51, 32, 40, 47, 53, 36, 35, 58, 59, 50, 25, 20, 14, 12, 20, 5, 29, 27,
          8, 7]
    }

df = DataFrame(Data, columns=['x', 'y'])

kmeans = KMeans(n_clusters=3).fit(df)
centroids = kmeans.cluster_centers_
print(centroids)

plt.scatter(df['x'], df['y'], c=kmeans.labels_.astype(float), s=50, alpha=0.5)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)