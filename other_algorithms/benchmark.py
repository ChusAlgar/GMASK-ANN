from neighbors_utils import *


# Set var for experiments:
datasets = ['gaussian']
distances = ['euclidean', 'manhattan', 'chebyshev']
methods = ['PYNN', 'FLANN', 'MASK']
knn = [5, 10, 15]

# Set params for MASK (municipios: tg=60, nc=30 , gaussian: (tg=1000, nc=500) (tg=20000 , nc=10000))
tg = 20000
nc = 10000

# For each dataset, calculate recalls, store them and print graph
for da in datasets:
    da_recalls = []
    for di in distances:
        di_recall = []
        for m in methods:
            m_recall = []
            for k in knn:
                re = recall(da, di, m, k, tg, nc)

                m_recall.append(re)
            di_recall.append(m_recall)
        da_recalls.append(di_recall)

    # Show results on a graph
    print_recall(da, distances, methods, knn, da_recalls)

'''
indices_le, coords_le, dists_le = get_neighbors('gaussian', 'euclidean', 'Brute_Force', 5)
indices_mc, coords_mc, dists_mc = get_neighbors('gaussian', 'euclidean', 'MASK', 5)

print(coords_le)
print("cambio-------------------------------------")
print(coords_mc)
'''

