from neighbors_utils import *


# Set var for experiments:
knn = [5, 10, 15]
methods = ['FLANN', 'pynn']
datasets = ['municipios']
distances = ['euclidean']

for k in knn:
    for m in methods:
        for da in datasets:
            for di in distances:
                recall(k, m, da, di)

'''
indices_le, coords_le, dists_le = get_neighbors(5, 'Brute_Force', 'gaussian', 'euclidean')
indices_mc, coords_mc, dists_mc = get_neighbors(5, 'pynn', 'gaussian', 'euclidean')

print(indices_le)
print("cambio-------------------------------------")
print(indices_mc)
'''

