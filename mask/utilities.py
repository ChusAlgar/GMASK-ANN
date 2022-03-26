# coding=utf-8
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances

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
    #cont = 0
    for idordenado in range(len(vector_ordenado)):
        iddesordenado = 0
        encontrado = False
        while iddesordenado<(len(vector_desordenado)) and not(encontrado):
            if vector_ordenado[idordenado] == vector_desordenado[iddesordenado]:
                encontrado = True
                correspondencia.append((idordenado, iddesordenado))
            iddesordenado += 1
        #cont += 1
        #print("identifica_nube", cont)

    for pareja in correspondencia:
        ido = pareja[0]
        idd = pareja[1]
        idnube = int(ido//cant_ptos)
        nubes_puntos[idnube].append(idd)
        puntos_nube[idd] = idnube

    return nubes_puntos, puntos_nube, nnubes


def identifica_nube_opt(vector_desordenado):
    puntos_nube = np.zeros(len(vector_desordenado), dtype=int)
    cont = 0
    vector_resultado = []
    for elem in vector_desordenado:
        puntos_nube[cont]=elem[2]
        punto = (elem[0], elem[1])
        vector_resultado.append(punto)
        cont += 1

    return vector_resultado, puntos_nube

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


### OPERACIONES PARA LA REAGRUPACIÓN DE GRUPOS DESPUÉS DE LA 1º DECONSTRUCCIÓN ###

def myFunc(e):
  return len(e)

def busca_candidato(D, id_grupo):
    '''filaini = D[id_grupo, 0:id_grupo]
    filafin = D[id_grupo, (id_grupo+1):]
    if len(filaini) == 0:
        fila = filafin
    elif len(filafin) == 0:
        fila = filaini
    else:
        fila = filaini + filafin
    elem_min = fila.min()
    fila = fila.tolist()
    indice = fila.index(elem_min)'''
    fila = D[id_grupo, : ].tolist()
    filaOrd = sorted(fila)
    minimo = filaOrd[1]
    indice = fila.index(minimo)
    return indice

def hay_grupos_peq(vector, tam_grupo):
    bandera = False
    cont = 0
    for elem in vector:
        if (len(elem) < tam_grupo):
            cont += 1

    if cont == len(vector):
        bandera = True

    return bandera

def  cluster_points(points, number_of_clusters):
    '''This function should take a list of points (in two dimensions) and return a list of clusters,
    each of which is a list of points. For example, if you passed in [(0, 0), (-0.1, 0.1), (2,3), (2.1, 3)]
    with number_of_clusters set to 2, it should return [[(0, 0), (-0.1, 0.1)], [(2,3), (2.1, 3)]].'''

    model = KMeans(n_clusters=number_of_clusters, random_state=0)
    distMat = model.fit_transform(points)
    resultList = [[] for i in range(number_of_clusters)]
    for i, rowList in enumerate(distMat):
        minIndex = min(enumerate(rowList), key = lambda x: x[1])[0]
        resultList[minIndex].append(points[i])
    return resultList

def reagrupacion(vector, tam_grupo):

    # Primero: reagrupación de grupos pequeños (numero de puntos < tam_grupo)
    # Ordenamos el vector colocando en las primeras posiciones los grupos más pequeños
    vector.sort(key=myFunc)
    cont = 0
    # lista en la que vamos a meter un representante de cada grupo para calcular sus distancias
    representantes = []
    for elem in vector:
        if len(elem) < tam_grupo:
            kmeans = KMeans(n_clusters=1, random_state=0).fit(elem)
            representantes.append(kmeans.cluster_centers_)
            cont += 1

    # Calculamos la matriz de distancias de los representantes
    representantes = np.concatenate(representantes).ravel().tolist()
    representantes = np.array(representantes)
    representantes = representantes.reshape(cont, 2)
    D = pairwise_distances(representantes, metric='euclidean')

    # Empezando por el grupo mas pequeño, reagrupamos con el más cercano hasta obtener un grupo con numero de puntos
    # mayor que tam_grupo
    lista_agrupaciones = np.empty((len(representantes),), dtype=object)
    for id_grupo in range(len(representantes)):
        id_candidatos = np.argsort(D[id_grupo, : ])
        npuntos = len(vector[id_grupo])
        cont = 1
        id_agrupados = []
        while npuntos < tam_grupo:
            id = id_candidatos[cont]
            npuntos += len(vector[id])
            id_agrupados.append(id)
            cont += 1

        lista_agrupaciones[id_grupo]=id_agrupados

    # Contamos el número de grupos que tenemos que hacer (ngrupos) y guardamos sus índices en conj_agrupados
    ngrupos = 0
    conj_agrupados = [[]]*len(lista_agrupaciones)
    lista = [[0], lista_agrupaciones[0]]
    # Aplanamos la lista (quitamos una dimensión)
    lista = sum(lista, [])
    conj_agrupados[0]= lista
    for id in range(1, len(lista_agrupaciones)):
        if not(any(id in sublist for sublist in conj_agrupados)):
            cuenta = 0
            indices = lista_agrupaciones[id]
            for id_cand in range(len(indices)):
                elem_to_find = indices[id_cand]
                if any(elem_to_find in sublist for sublist in conj_agrupados):
                    cuenta += 1
            if cuenta == len(indices):
                # Sus cercanos están en conj_agrupados
                conj_agrupados[ngrupos].append(id)
            else:
                # Sus cercano ni él están en conj_agrupados, tenemos que meterlos todos
                ngrupos += 1
                lista = [[id], lista_agrupaciones[id]]
                lista = sum(lista, [])
                conj_agrupados[ngrupos] = lista


    vector_intermedio = []
    # Metemos los grupos que acabamos de hacer
    for id in range(ngrupos+1):
        agrupados = conj_agrupados[id]
        puntos = []
        for elem in agrupados:
            puntos = puntos + vector[elem]
        vector_intermedio.append(puntos)



    # Segundo: dividimos los grupos grandes (numero de puntos < 2*tam_grupo)
    vector_final = []
    # Metemos los grupos grandes que no hemos tocado
    for elem in vector:
        if (len(elem) > tam_grupo):
            if len(elem) > (2 * tam_grupo):
                ndivisiones = int(len(elem) / tam_grupo)
                # Con el KMeans formamos el ndivisiones grupos
                list_points = cluster_points(elem, ndivisiones)
                # Guardamos los puntos asociados a cada ndivision en vector
                for cluster in list_points:
                    vector_final.append(cluster)
            else:
                vector_final.append(elem)

    for elem in vector_intermedio:
        if (len(elem) > (2*tam_grupo)):
            ndivisiones = int(len(elem)/tam_grupo)
            # Con el KMeans formamos el ndivisiones grupos
            # kmeans = KMeans(n_clusters=ndivisiones, random_state=0).fit(elem)
            list_points =cluster_points(elem, ndivisiones)
            # Guardamos los puntos asociados a cada ndivision en vector
            for cluster in list_points:
                vector_final.append(cluster)
        else:
            vector_final.append(elem)

    return vector_final