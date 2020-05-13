import numpy as np

def split_seq(seq, size):
    """Divide una secuencia en trozos de tamaño size con solapamiento"""
    return [seq[i:i + size] for i in range(len(seq) - (size - 1))]


def seq_count(sec, clav):
    """Cuenta el número de veces que aparece clav en sec"""
    return sec.count(clav)


def busca_dist_menor(m):
    elem_min = m[0, 1:].min()
    filas, columnas = m.shape
    encontrado = False
    colum = 1
    while colum < columnas and not encontrado:
        if elem_min == m[0, colum]:
            encontrado = True
        else:
            colum += 1
    return colum


def argmin_diagonal_ignored(m):
    mask = np.ones(m.shape, dtype=bool)
    np.fill_diagonal(mask, 0)
    elem_min = m[mask].min()
    fila = 0
    filas, columnas = m.shape
    encontrado = False
    while fila < filas and not encontrado:
        columna = fila + 1
        while columna < columnas and not encontrado:
            if elem_min == m[fila, columna]:
                encontrado = True
            columna += 1
        fila += 1
    return fila - 1, columna - 1


def kMedoids(D, k, tmax=100):
    # determine dimensions of distance matrix D
    m, n = D.shape

    if k > n:
        raise Exception('too many medoids')

    # find a set of valid initial cluster medoid indices since we
    # can't seed different clusters with two points at the same location
    valid_medoid_inds = set(range(n))
    invalid_medoid_inds = set([])
    rs, cs = np.where(D == 0)
    # the rows, cols must be shuffled because we will keep the first duplicate below
    index_shuf = list(range(len(rs)))
    np.random.shuffle(index_shuf)
    rs = rs[index_shuf]
    cs = cs[index_shuf]
    for r, c in zip(rs, cs):
        # if there are two points with a distance of 0...
        # keep the first one for cluster init
        if r < c and r not in invalid_medoid_inds:
            invalid_medoid_inds.add(c)
    valid_medoid_inds = list(valid_medoid_inds - invalid_medoid_inds)

    if k > len(valid_medoid_inds):
        raise Exception('too many medoids (after removing {} duplicate points)'.format(
            len(invalid_medoid_inds)))

    # randomly initialize an array of k medoid indices
    M = np.array(valid_medoid_inds)
    np.random.shuffle(M)
    M = np.sort(M[:k])

    # create a copy of the array of medoid indices
    Mnew = np.copy(M)

    # initialize a dictionary to represent clusters
    C = {}
    for t in range(tmax):
        # determine clusters, i. e. arrays of data indices
        J = np.argmin(D[:, M], axis=1)
        for kappa in range(k):
            C[kappa] = np.where(J == kappa)[0]
        # update cluster medoids
        for kappa in range(k):
            J = np.mean(D[np.ix_(C[kappa], C[kappa])], axis=1)
            j = np.argmin(J)
            Mnew[kappa] = C[kappa][j]
        np.sort(Mnew)
        # check for convergence
        if np.array_equal(M, Mnew):
            break
        M = np.copy(Mnew)
    else:
        # final update of cluster memberships
        J = np.argmin(D[:, M], axis=1)
        for kappa in range(k):
            C[kappa] = np.where(J == kappa)[0]

    # return results
    return M, C

def identifica_nube(vector_ordenado, vector_desordenado):
    nnubes = 8
    cant_ptos = int(len(vector_ordenado)//nnubes)
    nubes_puntos = [[] for i in range(nnubes)]
    puntos_nube = np.zeros(len(vector_desordenado), dtype=int)
    correspondencia = []
    for idordenado in range(len(vector_ordenado)):
        iddesordenado = 0
        encontrado = False
        while iddesordenado<(len(vector_desordenado)) and not(encontrado):
            if vector_ordenado[idordenado] == vector_desordenado[iddesordenado]:
                encontrado = True
                correspondencia.append((idordenado, iddesordenado))
            iddesordenado += 1

    for pareja in correspondencia:
        ido = pareja[0]
        idd = pareja[1]
        idnube = int(ido//cant_ptos)
        nubes_puntos[idnube].append(idd)
        puntos_nube[idd] = idnube

    return nubes_puntos, puntos_nube, nnubes

def obten_num_ptos(list_nptos, ncentroides):
    num_ptos = 0
    for elem in list_nptos:
        if elem > ncentroides:
            num_ptos+=ncentroides
        else:
            num_ptos+=elem
    return num_ptos

def divide_en_grupos(vector, longitud, ngrupos, tam):
    # longitud, _ = vector.shape
    if longitud == ngrupos*tam:
        # La división es exacta:
        vector = np.array(np.split(vector,ngrupos))
    else:
        # La división no es exacta:
        v1 = vector[:tam*(ngrupos-1)]
        resto = vector[tam*(ngrupos-1):]
        v1 = np.split(v1, ngrupos-1)
        v1.append(resto)
        vector = np.array(v1)
        #vector = np.concatenate([v1, resto])

    return vector

def obten_idgrupo(id_punto, grupos):
    ngrupos = len(grupos)
    id_grupo = 0
    inicio = 0
    fin = grupos[0]
    encontrado = False
    while (id_grupo < ngrupos) and not(encontrado):
        if id_punto>=inicio and id_punto<fin:
            encontrado = True
        else :
            id_grupo += 1
            inicio = fin
            fin = fin+grupos[id_grupo]

    return id_grupo