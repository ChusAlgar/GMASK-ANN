import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix
from sklearn.utils import shuffle


indptr = np.array([0, 2, 3, 6])
indices = np.array([0, 2, 2, 0, 1, 2])
data = np.array([1, 2, 3, 4, 5, 6])
matriz = csr_matrix((data, indices, indptr), shape=(3, 3))  #.toarray()

docs, terms = csr_matrix.get_shape(matriz)
print(docs, terms)

print("matris ordenada")
for d in range(0,docs):
    for t in matriz[d]:
        print(t)


print("matriz desordeanda")
matriz_ordenado = matriz.copy()
matriz_desordenado = shuffle(matriz_ordenado)

# vind = np.linspace(0, docs-1, docs, dtype=int)
# np.random.shuffle(vind)
# print("Vector de índcies fila: ",vind)
# cont = 0
#
# for elem in vind:
#     matriz_desordenado[cont] = matriz_ordenado[elem]
#     cont += 1

for d in range(0,docs):
    for t in matriz_desordenado[d]:
        print(t)

#M =(sparse.rand(10,3,.3,'csr')*10).astype(int)
#print(np.array_split(M.A, 3))
vector = np.array_split(matriz_desordenado.A, 2)
print(vector)