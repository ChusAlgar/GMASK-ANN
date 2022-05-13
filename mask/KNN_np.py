from pyclustering.utils.metric import distance_metric, type_metric
from pyclustering.cluster.kmeans import kmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from sklearn import preprocessing
import mask.utilities as util
from timeit import default_timer as timer
import logging
import mask.distances as dist


logger = logging.getLogger(__name__)


def calculate_numcapas(cant_ptos, tam_grupo, n_centroides):
    if cant_ptos < tam_grupo or tam_grupo == n_centroides:
        ncapas = 1
    else:
        cociente = int(cant_ptos / tam_grupo)
        resto = cant_ptos % tam_grupo
        grupos = cociente + resto
        new_ptos = grupos * n_centroides
        ncapas = 1
        while new_ptos > n_centroides:
            cociente = int(new_ptos / tam_grupo)
            resto = new_ptos % tam_grupo
            if resto == 0:
                grupos = cociente
                new_ptos = grupos * n_centroides
            elif resto < n_centroides:
                new_ptos = (cociente * n_centroides) + resto
                grupos = cociente + 1
            elif resto >= n_centroides:
                grupos = cociente + 1
                new_ptos = grupos * n_centroides

            if new_ptos >= n_centroides:
                ncapas += 1

    return ncapas


def built_estructuras_capa(cant_ptos, tam_grupo, n_centroides, n_capas):
    labels_capa = np.empty(n_capas, object)
    puntos_capa = np.empty(n_capas, object)
    grupos_capa = np.empty(n_capas, object)

    # Numero de grupos de la capa 0
    ngrupos = int(cant_ptos / tam_grupo)
    resto = cant_ptos % tam_grupo

    for capa in range(n_capas):

        if resto != 0:
            # resto = cant_ptos - (ngrupos * tam_grupo)
            ngrupos = ngrupos + 1
            labels_grupo = np.empty(ngrupos, object)
            for num in range(ngrupos - 1):
                labels_grupo[num] = np.zeros(tam_grupo, dtype=int)
            labels_grupo[ngrupos - 1] = np.zeros(resto, dtype=int)
            labels_capa[capa] = labels_grupo
            if (resto >= n_centroides):
                puntos_grupo = np.zeros((ngrupos, n_centroides, 2), dtype=float)
                resto_nuevo = (ngrupos * n_centroides) % tam_grupo
                ngrupos_nuevo = int((ngrupos * n_centroides) / tam_grupo)
            else:
                puntos_grupo = np.empty(ngrupos, object)
                for num in range(ngrupos - 1):
                    puntos_grupo[num] = np.zeros((ngrupos - 1, n_centroides, 2))
                puntos_grupo[ngrupos - 1] = np.zeros((1, resto, 2))
                resto_nuevo = ((ngrupos - 1) * n_centroides + resto) % tam_grupo
                ngrupos_nuevo = int(((ngrupos - 1) * n_centroides + resto) / tam_grupo)
            puntos_capa[capa] = puntos_grupo
            grupos_capa[capa] = np.zeros(ngrupos, dtype=int)
        else:
            puntos_capa[capa] = np.zeros((ngrupos, n_centroides, 2), dtype=float)
            labels_capa[capa] = np.zeros((ngrupos, tam_grupo), dtype=int)
            grupos_capa[capa] = np.zeros(ngrupos, dtype=int)
            resto_nuevo = (ngrupos * n_centroides) % tam_grupo
            ngrupos_nuevo = int((ngrupos * n_centroides) / tam_grupo)

        resto = resto_nuevo
        ngrupos = ngrupos_nuevo

    return puntos_capa, labels_capa, grupos_capa


# (nclouds, npc, tam_grupo, n_centroides, overlap):
def kmeans_tree(cant_ptos, tam_grupo, n_centroides, metrica, vector_original):

    # Parámetros de entada:
    # tam_grupo = tamaño del grupo para bombardear con los centroides (depende de la capacidad computacional).
    # n_centroides = número de centroides con los que se bombardea cada grupo
    # npc = número de puntos de cada nube
    opcion = 'kmeans'
    normaliza = False

    if metrica == 'euclidean':
        metric = distance_metric(type_metric.EUCLIDEAN)  # EUCLIDEAN, CHEBYSHEV, MANHATTAN)
    elif metrica == 'chebyshev':
        metric = distance_metric(type_metric.CHEBYSHEV)
    elif metrica == 'manhattan':
        metric = distance_metric(type_metric.MANHATTAN)
    elif metrica == 'user':
        metric = distance_metric(type_metric.USER_DEFINED, func=dist.euclidean)

    #    cant_ptos = nclouds * npc

        # Inicio del proceso iterativo de construcción-deconstrucción.
    start_time_constr = timer()

    vector = vector_original
    #    for iter in range(1):
    if normaliza:
        vector = preprocessing.normalize(vector, axis=0, norm='l2')

    # 23-03-2022
    print("calculo del número de capas")
    n_capas = calculate_numcapas(cant_ptos, tam_grupo, n_centroides)

    print("calculo de las estructuras de almacenamiento")
    puntos_capa, labels_capa, grupos_capa = built_estructuras_capa(cant_ptos, tam_grupo, n_centroides, n_capas)


    # Proceso iterativo para aplicar el kmeans o el kmedoids:
    print("INICIO PROCESO CONSTRUCCIÓN")
    for id_capa in range(n_capas):
        # Capa n:
        ngrupos = len(grupos_capa[id_capa])
        inicio = 0
        # 18-03-2021 puntos_grupo y labels_grupo ahora van a ser un np.array de tres dimensiones y los calculo
        # cuando calculo el número de grupos
        # puntos_grupo = []
        # labels_grupo = []
        cont_ptos = 0  # 03-03-2021. Contador de los puntos en cada capa
        # 23-03-2022    npuntos = []
        npuntos = np.zeros(ngrupos, dtype=int)
        for id_grupo in range(ngrupos):
            fin = inicio + tam_grupo
            # Inicio 03-03-2021. Control del último grupo (no tiene cantidad de puntos suficientes para formar
            # grupo
            if fin > cant_ptos:
                fin = cant_ptos
            # Fin 03-03-2021

            npuntos[id_grupo] = fin - inicio
            if ((fin - inicio) >= n_centroides):
                if opcion == 'kmeans':
                    # PYCLUSTERING
                    initial_centers = kmeans_plusplus_initializer(vector[inicio:fin], n_centroides).initialize()
                    kmeans_instance = kmeans(vector[inicio:fin], initial_centers, metric=metric)
                    kmeans_instance.process()
                    # 23-03-2022    puntos_grupo[id_grupo] = kmeans_instance.get_centers()
                    puntos_capa[id_capa][id_grupo] = kmeans_instance.get_centers()
                    clusters = kmeans_instance.get_clusters()
                    for num in range(fin - inicio):
                        not_find = True
                        sublist = 0
                        while not_find:
                            if num in clusters[sublist]:
                                # 23-03-2022    labels_grupo[id_grupo][num] = sublist
                                labels_capa[id_capa][id_grupo][num] = sublist
                                not_find = False
                            sublist += 1

                    # SKLEARN
                    # kmeans = KMeans(n_clusters=n_centroides, algorithm="full").fit(vector[inicio:fin])
                    # puntos_capa[id_capa][id_grupo] = kmeans.cluster_centers_
                    # labels_capa[id_capa][id_grupo] = kmeans.labels_

                    cont_ptos += n_centroides  # 03-03-20021

                else:
                    data = vector[inicio:fin]
                    D = pairwise_distances(data, metric=metrica)
                    M, C = util.kMedoids(D, n_centroides)
                    list_centroides = []
                    for point_idx in M:
                        list_centroides.append(data[point_idx])
                        # 23-03-2022    puntos_grupo.append(np.array(list_centroides))
                        puntos_capa[id_capa][id_grupo] = np.array(list_centroides)
                        cont_ptos += n_centroides  # 03-03-20021
                        # 23-03-2022    labels_grupo.append(M)
                        labels_capa[id_capa][id_grupo] = M

            else:
                # Si los puntos que tenemos en el grupo no es mayor que el número de centroides, no hacemos culster
                # 03-03-2021  puntos_grupo.append(vector[inicio:fin])  # aquí tenemos almacenados los puntos de la
                # siguiente capa para cada grupo
                # 18-03-2021 puntos_grupo.append(np.array(vector[inicio:fin]))  # aquí tenemos almacenados los puntos de la
                # 23-03-2022 puntos_grupo[id_grupo] = np.array(vector[inicio:fin])
                puntos_capa[id_capa][id_grupo] = np.array(vector[inicio:fin])
                # siguiente capa para cada grupo
                cont_ptos = cont_ptos + (fin - inicio)  # 03-03-2021
                etiquetas = []
                for i in range((fin - inicio)):
                    etiquetas.append(i)
                # 18-03-2021 labels_grupo.append(np.array(etiquetas))
                # 23-03-2022 labels_grupo[id_grupo] = np.array(etiquetas)
                labels_capa[id_capa][id_grupo] = np.array(etiquetas)

            inicio = fin

        # 23-03-2022    grupos_capa.append(npuntos)
        grupos_capa[id_capa] = npuntos

        # Guardamos los centroides de la capa para poder hacer el proceso inverso
        # 18-03-2021 array = np.array(puntos_grupo)
        # 18-03-2021 puntos_capa.append(array)
        # 23-03-2022    puntos_capa.append(puntos_grupo)
        # Guardamos las etiquetas de los puntos (índice del centroide con el que están asociados):
        # 18-03-2021 array2 = np.array(labels_grupo)
        # 18-03-2021 labels_capa.append(array2)
        # 23-03-2022    labels_capa.append(labels_grupo)
        # Almacenamos en vector los puntos de la siguiente capa
        # 03-03-2021 vector = array.reshape(-1, array.shape[-1])
        # 18-03-2021 vector = np.concatenate(puntos_grupo).ravel().tolist()  # 03-03-2021
        # 18-03-2021 vector = np.array(vector)
        # 18-03-2021 vector = vector.reshape(cont_ptos, 2)

        # vector = puntos_grupo
        # vector = vector.reshape(-1, vector.shape[-1])
        # vector = np.concatenate(puntos_grupo).ravel().tolist()  # 03-03-2021
        # vector = np.array(vector)
        vector = puntos_capa[id_capa]
        vector = np.concatenate(vector).ravel().tolist()  # 03-03-2021
        vector = np.array(vector)
        vector = vector.reshape(cont_ptos, 2)

        # Calculamos el numero de grupos de la siguiente capa
        # ngrupos = int(cont_ptos / tam_grupo)  # 03-03-2021  nfilas, ncolumnas = vector.shape
        # if ngrupos != 0:
        #     if (cont_ptos % tam_grupo) != 0:  # 03-03-2021 if (nfilas % tam_grupo) != 0:
        #         resto = cont_ptos - (ngrupos * tam_grupo)
        #         ngrupos = ngrupos + 1
        #         labels_grupo = np.empty(ngrupos, object)
        #         for num in range(ngrupos-1):
        #             labels_grupo[num] = np.zeros(tam_grupo, dtype=int)
        #         labels_grupo[ngrupos-1] = np.zeros(resto, dtype=int)
        #         if (resto >= n_centroides):
        #             puntos_grupo = np.zeros((ngrupos, n_centroides, 2), dtype=float)
        #         else:
        #             puntos_grupo = np.empty(ngrupos, object)
        #             for num in range(ngrupos - 1):
        #                 puntos_grupo[num] = np.zeros((ngrupos-1, n_centroides,2), dtype=float)
        #             puntos_grupo[ngrupos - 1] = np.zeros((1,resto,2), dtype=float)
        #     else:
        #         puntos_grupo = np.zeros((ngrupos, n_centroides, 2), dtype=float)
        #         labels_grupo = np.zeros((ngrupos, tam_grupo), dtype=int)
        #     # grupos_capa.append(ngrupos)
        cant_ptos = cont_ptos  # 03-03-2021 Actualizamos cant_ptos con el número de puntos del siguiente nivel
        # id_capa += 1

    print("FIN PROCESO CONSTRUCCIÓN")

    # 23-03-2022    n_capas = id_capa - 1
    end_time_constr = timer()
    # print("--- %s seconds ---", end_time_constr-start_time_constr)
    logger.info('tree time=%s seconds', end_time_constr - start_time_constr)

    return n_capas, grupos_capa, puntos_capa, labels_capa


def built_lista_pos(id_grupo, grupos_capa_compress, lista_pos):
    desplaz = 0
    for id in range(id_grupo):
        desplaz += grupos_capa_compress[id]
    result = lista_pos + desplaz
    return result


def search_near_centroid(id_grupo, id_centroide, id_ult_vecino, centroides_examinados, puntos_capa, labels_capa,
                         grupos_capa, ids_vecinos, vector_original, metrica):
    D = pairwise_distances(puntos_capa[0][id_grupo], metric=metrica)
    # min1 = D[id_centroide, 0:id_centroide].min()
    menor = np.partition(D[id_centroide], 1)[1]
    new_id_centroide = (np.argwhere(D[id_centroide] == menor)).ravel()

    if centroides_examinados[id_grupo][new_id_centroide] == 0:
        lista_pos = np.argwhere(labels_capa[0][id_grupo][:] == new_id_centroide)
        lista_pos = built_lista_pos(id_grupo, grupos_capa[0][:], lista_pos)
        lista_pos = lista_pos.ravel()

        new_lista_pos = np.setdiff1d(lista_pos, ids_vecinos)
        if len(new_lista_pos) >= 1:
            puntos_seleccionados = np.array(vector_original[new_lista_pos])
            vecino_guardado = (np.array(vector_original[id_ult_vecino])).reshape(1, 2)
            puntos_dist = np.concatenate([vecino_guardado, puntos_seleccionados])
            D = pairwise_distances(puntos_dist, metric=metrica)
            new_colum = util.busca_dist_menor(D)
            id_punto = new_lista_pos[new_colum - 1]
            dist = D[0, new_colum]

            if len(new_lista_pos) == 1:
                # No hay más puntos asociados con ese centroide, por lo que lo marcamos (ponemos su posición
                # a 1) como ya examinado
                centroides_examinados[id_grupo][new_id_centroide] = 1
        else:
            # Este centroide no nos vale, necesitamos otro:
            siguiente = 2
            salir = False
            while (centroides_examinados[id_grupo][new_id_centroide] == 1) and (siguiente < len(D[id_centroide]))\
                    and (not salir):
                menor = np.partition(D[id_centroide], siguiente)[siguiente]
                new_id_centroide = (np.argwhere(D[id_centroide] == menor)).ravel()

                lista_pos = np.argwhere(labels_capa[0][id_grupo][:] == new_id_centroide)
                lista_pos = built_lista_pos(id_grupo, grupos_capa[0][:], lista_pos)
                lista_pos = lista_pos.ravel()

                new_lista_pos = np.setdiff1d(lista_pos, ids_vecinos)
                if len(new_lista_pos) >= 1:
                    puntos_seleccionados = np.array(vector_original[new_lista_pos])
                    vecino_guardado = (np.array(vector_original[id_ult_vecino])).reshape(1, 2)
                    puntos_dist = np.concatenate([vecino_guardado, puntos_seleccionados])
                    D = pairwise_distances(puntos_dist, metric=metrica)
                    new_colum = util.busca_dist_menor(D)
                    id_punto = new_lista_pos[new_colum - 1]
                    dist = D[0, new_colum]
                    salir = True
                    if len(new_lista_pos) == 1:
                        # No hay más puntos asociados con ese centroide, por lo que lo marcamos (ponemos su posición
                        # a 1) como ya examinado
                        centroides_examinados[id_grupo][new_id_centroide] = 1
                else:
                    siguiente += 1

    else:
        # Si el centroide ha sido examinado, tengo que buscar el siguiente más cercano
        siguiente = 2
        maximo = len(D[id_centroide])
        salir = False
        while (centroides_examinados[id_grupo][new_id_centroide] == 1) and (siguiente < maximo) \
                and (not salir):
            menor = np.partition(D[id_centroide], siguiente)[siguiente]
            new_id_centroide = (np.argwhere(D[id_centroide] == menor)).ravel()

            lista_pos = np.argwhere(labels_capa[0][id_grupo][:] == new_id_centroide)
            lista_pos = built_lista_pos(id_grupo, grupos_capa[0][:], lista_pos)
            lista_pos = lista_pos.ravel()

            new_lista_pos = np.setdiff1d(lista_pos, ids_vecinos)
            if len(new_lista_pos) >= 1:
                puntos_seleccionados = np.array(vector_original[new_lista_pos])
                vecino_guardado = (np.array(vector_original[id_ult_vecino])).reshape(1, 2)
                puntos_dist = np.concatenate([vecino_guardado, puntos_seleccionados])
                D = pairwise_distances(puntos_dist, metric=metrica)
                new_colum = util.busca_dist_menor(D)
                id_punto = new_lista_pos[new_colum - 1]
                dist = D[0, new_colum]
                salir = True
                if len(new_lista_pos) == 1:
                    # No hay más puntos asociados con ese centroide, por lo que lo marcamos (ponemos su posición
                    # a 1) como ya examinado
                    centroides_examinados[id_grupo][new_id_centroide] = 1
            else:
                siguiente += 1


    return id_punto, dist, id_grupo, new_id_centroide


def kmeans_search(n_capas, n_centroides, seq_buscada, vector_original, vecinos, centroides_examinados,
                  n, metrica, grupos_capa, puntos_capa, labels_capa):
    print("********************PROCESO DECONSTRUCCIÓN*********************")
    # start_time_deconstr = timer()
#   n_capas = id_capa - 1
    logger.info('tree-depth=%s', n_capas)

    # lcorrespond = []
    # aciertos = 0
    # fallos = 0
    # vector_aux = []
    # vector_aux = np.empty((len(vector_original), 2), float)
    lista_pos = np.empty(100, int)
    # for i in range(len(vector_original)):
    # print('buscando punto ', i)
    # seq_buscada = np.array(vector_original[i])
    # seq_buscada = np.reshape(seq_buscada, (1, 2))

    seq_buscada = np.reshape(seq_buscada, (1, 2))
    for id_capa in range(n_capas-1, -1, -1):
        # 03-03-2021 Obtenemos los centroides de la capa
        centroides = puntos_capa[id_capa]
        centroides = np.concatenate(centroides) #.ravel() #.tolist()

        # 23-03-2022    if id_capa < n_capas:
        if id_capa < (n_capas - 1):
            # seleccionamos solo los puntos que están asociados con ese centroide
            centroides = np.array(centroides[lista_pos])

        puntos_dist = np.concatenate([seq_buscada, centroides])
        D = pairwise_distances(puntos_dist, metric=metrica)     # euclidean, chebyshev, manhattan
        columna = util.busca_dist_menor(D)
        # Corrección del índice del centroide
        # 23-03-2022    if id_capa != n_capas:
        if id_capa != (n_capas - 1):
            pos_centroide = lista_pos[columna - 1]
            if pos_centroide >= n_centroides:
                id_grupo = int(pos_centroide / n_centroides)
                id_centroide = pos_centroide - (id_grupo * n_centroides)
            else:
                id_centroide = pos_centroide
                id_grupo = 0
        else:
            # 08-03-2021. Corrección para cuando la última capa del arbol tiene más de un grupo
            if len(grupos_capa[id_capa]) > 1:
                if (columna - 1) >= n_centroides:
                    id_grupo = int((columna - 1) / n_centroides)
                    id_centroide = (columna - 1) - (id_grupo * n_centroides)
                else:
                    id_centroide = columna - 1
                    id_grupo = 0
                # 08-03-2021. Fin.
            else:
                id_centroide = columna - 1
                id_grupo = 0

        lista_pos_aux = np.argwhere(labels_capa[id_capa][id_grupo][:] == id_centroide)
        lista_pos = built_lista_pos(id_grupo, grupos_capa[id_capa][:], lista_pos_aux)
        lista_pos = lista_pos.ravel()

    # Capa de los datos:
    puntos_seleccionados = np.array(vector_original[lista_pos])
    puntos_dist = np.concatenate([seq_buscada, puntos_seleccionados])
    D = pairwise_distances(puntos_dist, metric=metrica)
    columna = util.busca_dist_menor(D)
    id_punto = lista_pos[columna - 1]

    # Control de los vecinos guardados (ciclados)
    # Si el id_punto encontrado ya lo teniamos guardado en vecinos, nos quedamos con el siguiente
    # mas cercano
    vecino = np.empty(5, object)
    almacenado = False
    if n == 0:
        # Guardamos directamente el vecino encontrado (es el primero)
        vecino[0] = id_punto
        vecino[1] = D[0, columna]
        vecino[2] = vector_original[id_punto]
        vecino[3] = id_grupo
        vecino[4] = id_centroide
        vecinos[n] = vecino
        almacenado = True
        if len(lista_pos) == 1:
            # No hay más puntos asociados con ese centroide, por lo que lo marcamos (ponemos su posición
            # a 1) como ya examinado
            centroides_examinados[id_grupo][id_centroide] = 1
    else:
        # Buscamos si el nuevo vecino esta ya guardado
        ids_vecinos = np.zeros(n, dtype=int)
        for x in range(n):
            ids_vecinos[x] = vecinos[x][0]
        index = np.ravel(np.asarray(ids_vecinos == id_punto).nonzero())

        if len(index) == 0:
            # No lo tenemos guardado, por lo tanto, lo guardamos
            vecino[0] = id_punto
            vecino[1] = D[0, columna]
            vecino[2] = vector_original[id_punto]
            vecino[3] = id_grupo
            vecino[4] = id_centroide
            vecinos[n] = vecino
            almacenado = True
            if len(lista_pos) == 1:
                # No hay más puntos asociados con ese centroide, por lo que lo marcamos (ponemos su posición
                # a 1) como ya examinado
                centroides_examinados[id_grupo][id_centroide] = 1
        else:
            # Si lo tenemos guardado. Buscamos otro candidato
            if len(lista_pos) == 1:
                # No tengo más candidatos asociados a ese centriode. Hay que buscar un nuevo centroide y examinar
                # sus candidatos
                id_ult_vecino = vecinos[n-1][0]
                id_punto, dist, id_grupo, new_id_centroide = search_near_centroid(id_grupo, id_centroide,
                                                            id_ult_vecino, centroides_examinados, puntos_capa,
                                                            labels_capa, grupos_capa, ids_vecinos,
                                                            vector_original, metrica)
                vecino[0] = id_punto
                vecino[1] = dist
                vecino[2] = vector_original[id_punto]
                vecino[3] = id_grupo
                vecino[4] = new_id_centroide
                vecinos[n] = vecino
                almacenado = True
            else:
                # Tenemos más candidatos asociados a ese centroide. Buscamos el siguiente punto más cercano
                new_lista_pos = np.setdiff1d(lista_pos, ids_vecinos)
                if len(new_lista_pos) == 0:
                    id_ult_vecino = vecinos[n - 1][0]
                    id_punto, dist, id_grupo, new_id_centroide = search_near_centroid(id_grupo, id_centroide,
                                                                                      id_ult_vecino,
                                                                                      centroides_examinados,
                                                                                      puntos_capa, labels_capa,
                                                                                      grupos_capa, ids_vecinos,
                                                                                      vector_original, metrica)
                    vecino[0] = id_punto
                    vecino[1] = dist
                    vecino[2] = vector_original[id_punto]
                    vecino[3] = id_grupo
                    vecino[4] = new_id_centroide
                    vecinos[n] = vecino
                    almacenado = True
                else:
                    puntos_seleccionados = np.array(vector_original[new_lista_pos])
                    vecino_guardado = (np.array(vector_original[id_punto])).reshape(1,2)
                    puntos_dist = np.concatenate([vecino_guardado, puntos_seleccionados])
                    D = pairwise_distances(puntos_dist, metric=metrica)
                    new_colum = util.busca_dist_menor(D)
                    id_punto = new_lista_pos[new_colum - 1]

                    vecino[0] = id_punto
                    vecino[1] = D[0, new_colum]
                    vecino[2] = vector_original[id_punto]
                    vecino[3] = id_grupo
                    vecino[4] = id_centroide
                    vecinos[n] = vecino
                    almacenado = True
                    if len(new_lista_pos) == 1:
                        # No hay más puntos asociados con ese centroide, por lo que lo marcamos (ponemos su posición
                        # a 1) como ya examinado
                        centroides_examinados[id_grupo][id_centroide] = 1

            # if len(lista_pos) == 0:
            #     # No tenemos más candidatos. Buscamos el siguiente centroide más cercano
            #     D = pairwise_distances(puntos_capa[0][id_grupo], metric=metrica)
            #     min1 = D[id_centroide, 0:id_centroide].min()
            #     if (id_centroide + 1) == len(D):
            #         new_id_centroide = (np.argwhere(D[id_centroide] == min1)).ravel()
            #     else:
            #         min2 = D[id_centroide, (id_centroide + 1):].min()
            #         if min1 <= min2:
            #             new_id_centroide = (np.argwhere(D[id_centroide] == min1)).ravel()
            #         else:
            #             new_id_centroide = (np.argwhere(D[id_centroide] == min2)).ravel()
            #     lista_pos_aux = np.argwhere(labels_capa[0][id_grupo][:] == new_id_centroide)
            #     lista_pos = built_lista_pos(id_grupo, grupos_capa[0][:], lista_pos_aux)
            #     lista_pos = lista_pos.ravel()
            #
            # while (not almacenado) and len(lista_pos)>=1 :
            # # if len(lista_pos) > 1:
            #     # Tenemos otros puntos asociados con el mismo centroide. Miramos si alguno de ellos no está
            #     # guardado.
            #     new_lista_pos = np.setdiff1d(lista_pos, ids_vecinos)
            #     if len(new_lista_pos) >= 1:
            #         # Nos quedamos con el que tenga menor distancia
            #         puntos_seleccionados = np.array(vector_original[new_lista_pos])
            #         vecino_guardado = (np.array(vector_original[id_punto])).reshape(1,2)
            #         puntos_dist = np.concatenate([vecino_guardado, puntos_seleccionados])
            #         D = pairwise_distances(puntos_dist, metric=metrica)
            #         new_colum = util.busca_dist_menor(D)
            #         id_punto = new_lista_pos[new_colum - 1]
            #
            #         vecino[0] = id_punto
            #         vecino[1] = D[0, new_colum]
            #         vecino[2] = vector_original[id_punto]
            #         vecinos[n] = vecino
            #         almacenado = True
            #     else:
            #         # Si todos los candidatos están ya guardados, tenemos que buscar el siguiente centroide mas
            #         # cercano cuyos puntos asociados, haya al menos uno que no esté guardado
            #         # Si hay varios que están guardados, nos quedaremos con el de menor distancia
            #         D = pairwise_distances(puntos_capa[0][id_grupo], metric=metrica)
            #         min1 = D[id_centroide, 0:id_centroide].min()
            #         if (id_centroide+1) == len(D):
            #             new_id_centroide = (np.argwhere(D[id_centroide] == min1)).ravel()
            #         else:
            #             min2 = D[id_centroide, (id_centroide+1):].min()
            #             if min1 <= min2:
            #                 new_id_centroide = (np.argwhere(D[id_centroide] == min1)).ravel()
            #             else:
            #                 new_id_centroide = (np.argwhere(D[id_centroide] == min2)).ravel()
            #         lista_pos_aux = np.argwhere(labels_capa[0][id_grupo][:] == new_id_centroide)
            #         lista_pos = built_lista_pos(id_grupo, grupos_capa[0][:], lista_pos_aux)
            #         lista_pos = lista_pos.ravel()

    # if n != 0:
    #     # Comprobamos que no esté guardado el nuevo vecino que hemos encontrado (id_punto)
    #     ids_vecinos = np.zeros(n, dtype=int)
    #     for x in range(n):
    #         ids_vecinos[x] = vecinos[x][0]
    #     # esta = np.where(ids_vecinos == id_punto)
    #     index = np.ravel(np.asarray(ids_vecinos == id_punto).nonzero())
    #     # print(len(index))
    #     #if esta.size==0:
    #     #if esta[0] is None:
    #     # if not np.any(esta):
    #     #    print("vacio")
    #     # else:
    #     if len(index) > 0:
    #         if len(lista_pos)>1:
    #             # El mas cercano no nos vale porque ya lo teniamos guardado en vecinos
    #             # Buscamos otro punto, asignado al mismo centroide, que tenga la distancia menor la vecino
    #             # ya guardado
    #             new_lista_pos = np.setdiff1d(lista_pos, ids_vecinos)
    #             puntos_seleccionados = np.array(vector_original[new_lista_pos])
    #             vecino_guardado = (np.array(vector_original[id_punto])).reshape(1,2)
    #             puntos_dist = np.concatenate([vecino_guardado, puntos_seleccionados])
    #             D = pairwise_distances(puntos_dist, metric=metrica)
    #             new_colum = util.busca_dist_menor(D)
    #             id_punto = new_lista_pos[new_colum - 1]
    #             # fila1 = D[0, 1:columna]
    #             # if columna+1 == len(D):
    #             #     fila = fila1
    #             # else:
    #             #     fila2 = D[0, columna+1:]
    #             #     fila = np.concatenate([fila1, fila2])
    #             # new_colum = np.argmin(fila)
    #             # id_punto = lista_pos[new_colum]
    #
    #             vecino[0] = id_punto
    #             # vecino[1] = fila2[0]
    #             # vecino[1] = fila[new_colum]
    #             vecino[1] = D[0, new_colum]
    #             vecino[2] = vector_original[id_punto]
    #             vecinos[n] = vecino
    #             almacenado = True
    #         else:
    #             # Busco el centroide más cercano al identificado
    #             D = pairwise_distances(puntos_capa[0][id_grupo], metric=metrica)
    #             min1 = D[id_centroide, 0:id_centroide].min()
    #             if (id_centroide+1) == len(D):
    #                 new_id_centroide = (np.argwhere(D[id_centroide] == min1)).ravel()
    #             else:
    #                 min2 = D[id_centroide, (id_centroide+1):].min()
    #                 if min1 <= min2:
    #                     new_id_centroide = (np.argwhere(D[id_centroide] == min1)).ravel()
    #                 else:
    #                     new_id_centroide = (np.argwhere(D[id_centroide] == min2)).ravel()
    #             # new_id_centroide = new_id_centroide.ravel()
    #             lista_pos_aux = np.argwhere(labels_capa[0][id_grupo][:] == new_id_centroide)
    #             lista_pos = built_lista_pos(id_grupo, grupos_capa[0][:], lista_pos_aux)
    #             lista_pos = lista_pos.ravel()
    #
    #             # De todos los puntos asociados a ese centroide, me quedo con el que me de menos distancia
    #             puntos_seleccionados = np.array(vector_original[lista_pos])
    #             puntos_dist = np.concatenate([seq_buscada, puntos_seleccionados])
    #             D = pairwise_distances(puntos_dist, metric=metrica)
    #             columna = util.busca_dist_menor(D)
    #             id_punto = lista_pos[columna - 1]
    #
    #             # Lo guardamos:
    #             vecino[0] = id_punto
    #             vecino[1] = D[0, columna]
    #             vecino[2] = vector_original[id_punto]
    #             vecinos[n] = vecino
    #             almacenado = True
    #     else:
    #         # El vecino encontrado no lo teniamos guardado
    #         vecino[0] = id_punto
    #         vecino[1] = D[0, columna]
    #         vecino[2] = vector_original[id_punto]
    #         vecinos[n] = vecino
    #         almacenado = True
    # else:
    #     vecino[0] = id_punto
    #     vecino[1] = D[0, columna]
    #     vecino[2] = vector_original[id_punto]
    #     vecinos[n] = vecino
    #     almacenado = True

    print("FIN PROCESO DECONSTRUCCIÓN")
    # end_time_deconstr = timer()
    # print("--- %s seconds ---", end_time_deconstr-start_time_deconstr)
    # logger.info('search time= %s seconds', end_time_deconstr - start_time_deconstr)

    return almacenado