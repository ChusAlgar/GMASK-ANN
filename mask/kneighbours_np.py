from pyclustering.utils.metric import distance_metric, type_metric
from pyclustering.cluster.kmeans import kmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from sklearn import preprocessing
import mask.utilities as util
from timeit import default_timer as timer
import logging
import mask.kmeans_tree_npdist_modulado as ktd
import mask.distances as dist
import math
from sklearn.cluster import KMeans


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


def generalbuilt_estructuras_capa(vecordenacion, tam_grupo, n_centroides):
    labels_capa = np.empty(len(vecordenacion), object)
    puntos_capa = np.empty(len(vecordenacion), object)
    grupos_capa = np.empty(len(vecordenacion), object)


    for i in range(len(vecordenacion)):
        num = vecordenacion[i, 0]
        # ncapas = calculate_numcapas(num, tam_grupo, n_centroides)
        if num != 0:
            if num <= tam_grupo:
                ncapas = 1
            else:
                ncapas = int(num / tam_grupo)
                resto = num % tam_grupo
                if resto <= n_centroides:
                    ncapas += 1
                else:
                    ncapas += 2

            labels_capa[i] = np.empty(ncapas, object)
            puntos_capa[i] = np.empty(ncapas, object)
            grupos_capa[i] = np.empty(ncapas, object)

            # Numero de grupos de la capa 0
            ngrupos = int(num / tam_grupo)
            resto = num % tam_grupo

            for capa in range(ncapas):

                if resto != 0:
                    ngrupos = ngrupos + 1
                    labels_grupo = np.empty(ngrupos, object)
                    for num in range(ngrupos - 1):
                        labels_grupo[num] = np.zeros(tam_grupo, dtype=int)
                    labels_grupo[ngrupos - 1] = np.zeros(resto, dtype=int)
                    labels_capa[i][capa] = labels_grupo
                    if (resto >= n_centroides):
                        puntos_grupo = np.zeros((ngrupos, n_centroides, 2), dtype=float)
                        resto_nuevo = (ngrupos * n_centroides) % tam_grupo
                        ngrupos_nuevo = int((ngrupos * n_centroides) / tam_grupo)
                    else:
                        puntos_grupo = np.empty(ngrupos, object)
                        for nume in range(ngrupos - 1):
                            puntos_grupo[nume] = np.zeros((ngrupos - 1, n_centroides, 2))
                        puntos_grupo[ngrupos - 1] = np.zeros((1, resto, 2))
                        resto_nuevo = ((ngrupos - 1) * n_centroides + resto) % tam_grupo
                        ngrupos_nuevo = int(((ngrupos - 1) * n_centroides + resto) / tam_grupo)
                    puntos_capa[i][capa] = puntos_grupo
                    grupos_capa[i][capa] = np.zeros(ngrupos, dtype=int)
                else:
                    puntos_capa[i][capa] = np.zeros((ngrupos, n_centroides, 2), dtype=float)
                    labels_capa[i][capa] = np.zeros((ngrupos, tam_grupo), dtype=int)
                    grupos_capa[i][capa] = np.zeros(ngrupos, dtype=int)
                    resto_nuevo = (ngrupos * n_centroides) % tam_grupo
                    ngrupos_nuevo = int((ngrupos * n_centroides) / tam_grupo)

                resto = resto_nuevo
                ngrupos = ngrupos_nuevo

    return puntos_capa, labels_capa, grupos_capa


def kmeans_treeini(cant_ptos, tam_grupo, n_centroides, metrica,
                vector_original):  # (nclouds, npc, tam_grupo, n_centroides, overlap):
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


def kmeans_searchini(n_capas, n_centroides, vector_original, tam_grupo, metrica, grupos_capa, puntos_capa, labels_capa):
    print("********************PROCESO DECONSTRUCCIÓN*********************")
    start_time_deconstr = timer()
    #   n_capas = id_capa - 1
    logger.info('tree-depth=%s', n_capas)

    aciertos = 0
    fallos = 0
    # vector_aux = []
    vector_aux = np.empty((len(vector_original), 2), float)
    lista_pos = np.empty(100, int)
    ngrupos = len(grupos_capa[0])
    vecordenacion_aux = np.empty((ngrupos, 2), object)
    for j in range(ngrupos):
        vecordenacion_aux[j, 0] = 0
        vecordenacion_aux[j, 1] = []

    for i in range(len(vector_original)):
        # print('buscando punto ', i)
        seq_buscada = np.array(vector_original[i])
        seq_buscada = np.reshape(seq_buscada, (1, 2))

        for id_capa in range(n_capas - 1, -1, -1):
            # 18-03-2021 Ahora puntos_capa es una list de ndarrays
            # 03-03-2021 Obtenemos los centroides de la capa

            centroides = puntos_capa[id_capa]
            # 18-03-2021 centroides = np.array(puntos_capa[id_capa])
            # centroides = centroides.reshape(-1, centroides.shape[-1])
            # centroides = np.concatenate(centroides).ravel().tolist()
            centroides = np.concatenate(centroides)  # .ravel() #.tolist()
            # Fin 18-03-2021

            # 23-03-2022    if id_capa < n_capas:
            if id_capa < (n_capas - 1):
                # seleccionamos solo los puntos que están asociados con ese centroide
                centroides = np.array(centroides[lista_pos])

            puntos_dist = np.concatenate([seq_buscada, centroides])
            D = pairwise_distances(puntos_dist, metric=metrica)  # euclidean, chebyshev, manhattan
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
                # etiquetas = labels_capa[id_capa + 1]
                # etiquetas = etiquetas.reshape(-1, etiquetas.shape[-1])
                # id_centroide = etiquetas[0][pos_centroide]
            else:
                # 08-03-2021. Corrección para cuando la última capa del arbol tiene más de un grupo
                # 23-03-2022    if len(grupos_capa[n_capas]) > 1:
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
            # print(id_centroide)

            lista_pos_aux = np.argwhere(labels_capa[id_capa][id_grupo][:] == id_centroide)
            lista_pos = built_lista_pos(id_grupo, grupos_capa[id_capa][:], lista_pos_aux)
            lista_pos = lista_pos.ravel()

        # Capa de los datos:
        # Solo miramos la distancia con los que están en el mismo grupo
        puntos_seleccionados = np.array(vector_original[lista_pos])
        puntos_dist = np.concatenate([seq_buscada, puntos_seleccionados])
        D = pairwise_distances(puntos_dist, metric=metrica)
        columna = util.busca_dist_menor(D)
        id_punto = lista_pos[columna - 1]


        # vector_aux[i] = vector_original[id_punto]

        # Contamos los aciertos y los fallos
        grupo_pbuscado = int(i/tam_grupo)  #identifica_grupo(i, tam_grupo)
        grupo_pencontrado = int(id_punto/tam_grupo)    #identifica_grupo(id_punto, tam_grupo)
        if grupo_pbuscado == grupo_pencontrado:
            aciertos += 1
        else:
            fallos += 1
        vecordenacion_aux[grupo_pencontrado, 0] += 1
        listant = vecordenacion_aux[grupo_pencontrado, 1]
        listant.append(i)
        vecordenacion_aux[grupo_pencontrado, 1] = listant

    # vector = vector_aux

    # Limpiamos vecordenacion_aux
    contnoceros = 0
    for elem in vecordenacion_aux:
        if elem[0] != 0:
            contnoceros += 1
    vecordenacion= np.empty((contnoceros, 2), object)
    cont = 0
    for i in range(len(vecordenacion_aux)):
        if vecordenacion_aux[i, 0] != 0:
            vecordenacion[cont] = vecordenacion_aux[i]
            cont += 1


    print("FIN PROCESO DECONSTRUCCIÓN")
    end_time_deconstr = timer()
    # print("--- %s seconds ---", end_time_deconstr-start_time_deconstr)
    logger.info('search time= %s seconds', end_time_deconstr - start_time_deconstr)

    return aciertos, fallos, vecordenacion


def kmeans_search(n_capas, vector_original, vec_ordenacion_ant, tam_grupo, metrica, grupos_capa, puntos_capa, labels_capa):
    print("********************PROCESO DECONSTRUCCIÓN*********************")
    start_time_deconstr = timer()
    logger.info('tree-depth=%s', n_capas)

    aciertos = 0
    fallos = 0
    # vector_aux = np.empty((len(vector_original), 2), float)
    # lista_pos = np.empty(100, int)
    ngrupos = len(grupos_capa[0])
    vecordenacion_aux = np.empty((ngrupos, 2), object)
    for j in range(ngrupos):
        vecordenacion_aux[j, 0] = 0
        vecordenacion_aux[j, 1] = []

    for i in range(len(vector_original)):
        # print('buscando punto ', i)
        seq_buscada = np.array(vector_original[i])
        seq_buscada = np.reshape(seq_buscada, (1, 2))

        for id_capa in range(n_capas - 1, -1, -1):
            # 18-03-2021 Ahora puntos_capa es una list de ndarrays
            # 03-03-2021 Obtenemos los centroides de la capa

            centroides = puntos_capa[id_capa]
            # 18-03-2021 centroides = np.array(puntos_capa[id_capa])
            # centroides = centroides.reshape(-1, centroides.shape[-1])
            # centroides = np.concatenate(centroides).ravel().tolist()
            centroides = np.concatenate(centroides)  # .ravel() #.tolist()
            # Fin 18-03-2021

            # 23-03-2022    if id_capa < n_capas:
            if id_capa < (n_capas - 1):
                # seleccionamos solo los puntos que están asociados con ese centroide
                centroides = np.array(centroides[lista_pos])

            puntos_dist = np.concatenate([seq_buscada, centroides])
            D = pairwise_distances(puntos_dist, metric=metrica)  # euclidean, chebyshev, manhattan
            columna = util.busca_dist_menor(D)
            # Corrección del índice del centroide
            id_centroide = columna - 1
            id_grupo = 0
            if id_centroide >= len(puntos_capa[id_capa][id_grupo]):
                contador = len(puntos_capa[id_capa][id_grupo])
                while contador < columna:
                    id_grupo += 1
                    cont_ant = contador
                    contador += len(puntos_capa[id_capa][id_grupo])
                    id_centroide = (columna - 1) - cont_ant

            # 23-03-2022    if id_capa != n_capas:
            # if id_capa != (n_capas - 1):
            #     pos_centroide = lista_pos[columna - 1]
            #     if pos_centroide >= n_centroides:
            #         id_grupo = int(pos_centroide / n_centroides)
            #         id_centroide = pos_centroide - (id_grupo * n_centroides)
            #     else:
            #         id_centroide = pos_centroide
            #         id_grupo = 0
            #     # etiquetas = labels_capa[id_capa + 1]
            #     # etiquetas = etiquetas.reshape(-1, etiquetas.shape[-1])
            #     # id_centroide = etiquetas[0][pos_centroide]
            # else:
            #     # 08-03-2021. Corrección para cuando la última capa del arbol tiene más de un grupo
            #     # 23-03-2022    if len(grupos_capa[n_capas]) > 1:
            #     if len(grupos_capa[id_capa]) > 1:
            #         if (columna - 1) >= n_centroides:
            #             id_grupo = int((columna - 1) / n_centroides)
            #             id_centroide = (columna - 1) - (id_grupo * n_centroides)
            #         else:
            #             id_centroide = columna - 1
            #             id_grupo = 0
            #         # 08-03-2021. Fin.
            #     else:
            #         id_centroide = columna - 1
            #         id_grupo = 0


            lista_pos_aux = np.argwhere(labels_capa[id_capa][id_grupo][:] == id_centroide)
            # lista_pos = built_lista_pos(id_grupo, grupos_capa[id_capa][:], lista_pos_aux)
            # lista_pos = lista_pos.ravel()
            lista_pos_aux = lista_pos_aux.ravel()
            lista_pos = []
            for id in lista_pos_aux:
                idp = vec_ordenacion_ant[id_grupo, 1][id]
                lista_pos.append(idp)


        # Capa de los datos:
        # Solo miramos la distancia con los que están en el mismo grupo
        puntos_seleccionados = np.array(vector_original[lista_pos])
        puntos_dist = np.concatenate([seq_buscada, puntos_seleccionados])
        D = pairwise_distances(puntos_dist, metric=metrica)
        columna = util.busca_dist_menor(D)
        # id_punto = lista_pos[columna - 1]

        # vector_aux[i] = vector_original[id_punto]

        # Contamos los aciertos y los fallos
        grupo_pbuscado = int(i / tam_grupo)  # identifica_grupo(i, tam_grupo)
        grupo_pencontrado = id_grupo    #int(id_punto / tam_grupo)  # identifica_grupo(id_punto, tam_grupo)
        # if grupo_pbuscado == grupo_pencontrado:
        valor = D[0, columna]
        if D[0, columna] <= 1.0e-10:
            aciertos += 1
        else:
            fallos += 1
        vecordenacion_aux[grupo_pencontrado, 0] += 1
        listant = vecordenacion_aux[grupo_pencontrado, 1]
        listant.append(i)
        vecordenacion_aux[grupo_pencontrado, 1] = listant

    # vector = vector_aux

    # Limpiamos vecordenacion_aux
    contnoceros = 0
    for elem in vecordenacion_aux:
        if elem[0] != 0:
            contnoceros += 1
    vecordenacion = np.empty((contnoceros, 2), object)
    cont = 0
    for i in range(len(vecordenacion_aux)):
        if vecordenacion_aux[i, 0] != 0:
            vecordenacion[cont] = vecordenacion_aux[i]
            cont += 1

    print("FIN PROCESO DECONSTRUCCIÓN")
    end_time_deconstr = timer()
    # print("--- %s seconds ---", end_time_deconstr-start_time_deconstr)
    logger.info('search time= %s seconds', end_time_deconstr - start_time_deconstr)

    return aciertos, fallos, vecordenacion


def sort_distance(k_vecinos, id_punto, punto, vector_original, grupos_capa0, metrica):
    id_grupo = 0
    aux_idpunto = id_punto
    min = 0
    max = grupos_capa0[id_grupo]-1
    while aux_idpunto > grupos_capa0[id_grupo]:
        min = min + grupos_capa0[id_grupo]
        max = max + grupos_capa0[id_grupo]
        aux_idpunto = aux_idpunto-grupos_capa0[id_grupo]
        id_grupo += 1

    if metrica == 'euclidean':
        metric = distance_metric(type_metric.EUCLIDEAN)  # EUCLIDEAN, CHEBYSHEV, MANHATTAN)
    elif metrica == 'chebyshev':
        metric = distance_metric(type_metric.CHEBYSHEV)
    elif metrica == 'manhattan':
        metric = distance_metric(type_metric.MANHATTAN)
    elif metrica == 'user':
        metric = distance_metric(type_metric.USER_DEFINED, func=dist.euclidean)

    # # Opcion de calcular solo la distancia entre el punto que buscamos y los candidatos
    tamano = max - min + 1
    distances = np.empty((tamano, 2), object)
    # # distances = np.empty(tamano, object)
    # candidatos = vector_original[min:(max+1), :]
    # for i in range(tamano):
    #     distances[i, 0] = candidatos[i]
    #     # vecdist = metric(punto, candidatos[i])
    #     # distances[i, 1] = math.sqrt(vecdist[0]*vecdist[0]+vecdist[1]*vecdist[1]) #distances[i, 1] = np.linalg.norm(vecdist)
    #     # distances[i] = np.concatenate([candidatos[i],np.linalg.norm(vecdist)])
    #
    #     # Opcion cutre (funciona bien): calculo la distancia euclidea a mano
    #     vecdist = punto - candidatos[i]
    #     distances[i, 1] = math.sqrt(vecdist[0][0]*vecdist[0][0]+vecdist[0][1]*vecdist[0][1])

    # Opcion de obtener la matriz de distancia
    puntos_dist = np.concatenate([punto, vector_original[min:(max+1), :]])
    D = pairwise_distances(puntos_dist, metric=metrica)
    distances[:, 1] = D[0][1:]
    # candidatos = vector_original[min:(max + 1), :]
    # for i in range(tamano):
    #    distances[i, 0] = vector_original[min+i, :]
    for i in range(tamano):
        distances[i, 0] = min + i

    # distances = np.sort(distances, axis=1)  # ordenación por el segundo eje
    # result = distances[0:k_vecinos, :]
    # result = np.argpartition(-distances[..., -1].flatten(), k_vecinos)
    result = sorted(distances, key=lambda kv: kv[1])[0:k_vecinos]

    return result


def sort_distance_new(k_vecinos, id_grupo, punto, vector_original, vec_ordenacion, metrica):

    if metrica == 'euclidean':
        metric = distance_metric(type_metric.EUCLIDEAN)  # EUCLIDEAN, CHEBYSHEV, MANHATTAN)
    elif metrica == 'chebyshev':
        metric = distance_metric(type_metric.CHEBYSHEV)
    elif metrica == 'manhattan':
        metric = distance_metric(type_metric.MANHATTAN)
    elif metrica == 'user':
        metric = distance_metric(type_metric.USER_DEFINED, func=dist.euclidean)

    # Calculamos solo la distancia entre el punto que buscamos y los candidatos
    candidatos = vec_ordenacion[id_grupo, 1]
    tamano = len(candidatos)
    distances = np.empty((tamano, 2), object)
    coords_candidatos = vector_original[candidatos, :]

    # Opcion de obtener la matriz de distancia
    puntos_dist = np.concatenate([punto, coords_candidatos])
    D = pairwise_distances(puntos_dist, metric=metrica)
    distances[:, 1] = D[0][1:]
    distances[:, 0] = candidatos

    # distances = np.sort(distances, axis=1)  # ordenación por el segundo eje
    # result = distances[0:k_vecinos, :]
    # result = np.argpartition(-distances[..., -1].flatten(), k_vecinos)
    result = sorted(distances, key=lambda kv: kv[1])[0:k_vecinos]

    return result


# def kneighbours_search(k_vecinos, punto, n_capas, n_centroides, vector_original, metrica, grupos_capa, puntos_capa, labels_capa):
def kneighbours_search(k_vecinos, punto, n_capas, vec_ordenacion_ant, vector_original, metrica, grupos_capa, puntos_capa,
                           labels_capa):

    print("INICIO BÚSQUEDA K-VECINOS")
    start_time_deconstr = timer()
    logger.info('k-neighbours=%s', k_vecinos)

    for id_capa in range(n_capas - 1, -1, -1):
        centroides = puntos_capa[id_capa]
        centroides = np.concatenate(centroides)  # .ravel() #.tolist()

        if id_capa < (n_capas - 1):
            centroides = np.array(centroides[lista_pos])

        puntos_dist = np.concatenate([punto, centroides])
        D = pairwise_distances(puntos_dist, metric=metrica)  # euclidean, chebyshev, manhattan
        columna = util.busca_dist_menor(D)
        # Corrección del índice del centroide
        # if id_capa != (n_capas - 1):
        #     pos_centroide = lista_pos[columna - 1]
        #     if pos_centroide >= n_centroides:
        #         id_grupo = int(pos_centroide / n_centroides)
        #         id_centroide = pos_centroide - (id_grupo * n_centroides)
        #     else:
        #         id_centroide = pos_centroide
        #         id_grupo = 0
        # else:
        #     # 08-03-2021. Corrección para cuando la última capa del arbol tiene más de un grupo
        #     if len(grupos_capa[id_capa]) > 1:
        #         if (columna - 1) >= n_centroides:
        #             id_grupo = int((columna - 1) / n_centroides)
        #             id_centroide = (columna - 1) - (id_grupo * n_centroides)
        #         else:
        #             id_centroide = columna - 1
        #             id_grupo = 0
        #     else:
        #         id_centroide = columna - 1
        #         id_grupo = 0
        #
        # lista_pos_aux = np.argwhere(labels_capa[id_capa][id_grupo][:] == id_centroide)
        # lista_pos = ktd.built_lista_pos(id_grupo, grupos_capa[id_capa][:], lista_pos_aux)
        # lista_pos = lista_pos.ravel()

        id_centroide = columna - 1
        id_grupo = 0
        if id_centroide >= len(puntos_capa[id_capa][id_grupo]):
            contador = len(puntos_capa[id_capa][id_grupo])
            while contador < columna:
                id_grupo += 1
                cont_ant = contador
                contador += len(puntos_capa[id_capa][id_grupo])
                id_centroide = (columna - 1) - cont_ant
        lista_pos_aux = np.argwhere(labels_capa[id_capa][id_grupo][:] == id_centroide)
        # lista_pos = built_lista_pos(id_grupo, grupos_capa[id_capa][:], lista_pos_aux)
        # lista_pos = lista_pos.ravel()
        lista_pos_aux = lista_pos_aux.ravel()
        lista_pos = []
        for id in lista_pos_aux:
            idp = vec_ordenacion_ant[id_grupo, 1][id]
            lista_pos.append(idp)

    ''' # Capa anterior a los datos:
    puntos_seleccionados = np.array(vector_original[lista_pos])
    puntos_dist = np.concatenate([punto, puntos_seleccionados])
    D = pairwise_distances(puntos_dist, metric=metrica)
    columna = util.busca_dist_menor(D)
    # Corrección para cuando la última capa del arbol tiene más de un grupo
    if len(grupos_capa[1]) > 1:
        # Como estamos en la capa anterior a los datos, id_capa=1
        if (columna - 1) >= n_centroides:
            id_grupo = int((columna - 1) / n_centroides)
            id_centroide = (columna - 1) - (id_grupo * n_centroides)
        else:
            id_centroide = columna - 1
            id_grupo = 0
    else:
        id_centroide = columna - 1
        id_grupo = 0
    lista_pos_aux = np.argwhere(labels_capa[1][id_grupo][:] == id_centroide)
    lista_pos = ktd.built_lista_pos(id_grupo, grupos_capa[1][:], lista_pos_aux)
    lista_pos = lista_pos.ravel()'''

    # Capa de los datos:
    puntos_seleccionados = np.array(vector_original[lista_pos])
    puntos_dist = np.concatenate([punto, puntos_seleccionados])
    D = pairwise_distances(puntos_dist, metric=metrica)
    columna = util.busca_dist_menor(D)
    id_punto = lista_pos[columna - 1]

    # vecinos = sort_distance(k_vecinos, id_punto, punto, vector_original, grupos_capa[0], metrica)
    vecinos = sort_distance_new(k_vecinos, id_grupo, punto, vector_original, vec_ordenacion_ant, metrica)
    # vecinos = np.array([id_punto, D[0][columna]])

    print("FIN BÚSQUEDA K-VECINOS")
    end_time_deconstr = timer()
    logger.info('search neighbours time= %s seconds', end_time_deconstr - start_time_deconstr)

    return vecinos


def kmeans_tree(tam_grupo, n_centroides, metrica, vector_original, vecordenacion):
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

    # Inicio del proceso iterativo de construcción-deconstrucción.
    start_time_constr = timer()

    vector = vector_original
    if normaliza:
        vector = preprocessing.normalize(vector, axis=0, norm='l2')

    print("calculo de las estructuras de almacenamiento")
    # puntos_capa, labels_capa, grupos_capa = generalbuilt_estructuras_capa(vecordenacion, tam_grupo, n_centroides)
    n_capas = 1
    ngrupos = len(vecordenacion)
    labels_capa = np.empty(n_capas, object)
    puntos_capa = np.empty(n_capas, object)
    grupos_capa = np.empty(n_capas, object)
    n_cent_nuevo = np.zeros(ngrupos, dtype=int)
    for capa in range(n_capas):
        grupos_capa[capa] = np.zeros(ngrupos, dtype=int)
        labels_capa[capa] = np.empty(ngrupos, object)
        puntos_capa[capa] = np.empty(ngrupos, object)
        for grup in range(ngrupos):
            if vecordenacion[grup, 0] < tam_grupo:
                n_cent_nuevo[grup] = vecordenacion[grup, 0]
            elif (tam_grupo <= vecordenacion[grup, 0] < (2*tam_grupo)):
                n_cent_nuevo[grup] = n_centroides
            else:
                n_cent_nuevo[grup] = int(vecordenacion[grup, 0]/tam_grupo) * n_centroides
            puntos_capa[capa][grup] = np.zeros((n_cent_nuevo[grup], 2), dtype=float)
            # if vecordenacion[grup, 0] >= n_centroides:
            #     puntos_capa[capa][grup] = np.zeros((n_centroides, 2), dtype=float)
            # else:
            #     puntos_capa[capa][grup] = np.zeros((vecordenacion[grup, 0], 2), dtype=float)
            labels_capa[capa][grup] = np.zeros(vecordenacion[grup, 0], dtype=int)


    # labels_capa = np.empty((1,len(vecordenacion)), object)
    # puntos_capa = np.empty((1,len(vecordenacion)), object)
    # grupos_capa = np.empty((1,len(vecordenacion)), object)
    # for i in range(len(vecordenacion)):
    #     labels_capa[0, i] = np.zeros((vecordenacion[i, 0], 1), dtype=int)
    #     puntos_capa[0, i] = np.zeros((n_centroides, 2), dtype=float)
    #     grupos_capa[0, i] = np.zeros(1, dtype=int)


    # Proceso iterativo para aplicar el kmeans o el kmedoids:
    print("INICIO PROCESO CONSTRUCCIÓN")
    inicio = 0

    for id_capa in range(n_capas):
        # Capa n:
        # ngrupos = len(vecordenacion)  #len(grupos_capa[id0])

        # 18-03-2021 puntos_grupo y labels_grupo ahora van a ser un np.array de tres dimensiones y los calculo
        # cuando calculo el número de grupos
        cont_ptos = 0  # 03-03-2021. Contador de los puntos en cada capa
        # 23-03-2022    npuntos = []
        npuntos = np.zeros(ngrupos, dtype=int)
        for id_grupo in range(ngrupos):
            fin = inicio + vecordenacion[id_grupo, 0]
            # Inicio 03-03-2021. Control del último grupo (no tiene cantidad de puntos suficientes para formar
            # grupo
            # if fin > cant_ptos:
            #    fin = cant_ptos
            # Fin 03-03-2021

            npuntos[id_grupo] = fin - inicio
            if ((fin - inicio) >= n_centroides):
                if opcion == 'kmeans':
                    # PYCLUSTERING
                    # amount_candidates = kmeans_plusplus_initializer.FARTHEST_CENTER_CANDIDATE
                    # initial_centers = kmeans_plusplus_initializer(vector[inicio:fin], n_centroides, amount_candidates).initialize()
                    initial_centers = kmeans_plusplus_initializer(vector[inicio:fin], n_cent_nuevo[id_grupo]).initialize()
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
                    # kmeans = KMeans(n_clusters=n_cent_nuevo[id_grupo], algorithm="full").fit(vector[inicio:fin])
                    # puntos_capa[id_capa][id_grupo] = kmeans.cluster_centers_
                    # labels_capa[id_capa][id_grupo] = kmeans.labels_

                    cont_ptos += n_centroides  # 03-03-20021
                    # labels_capa[id_capa][id_grupo] = np.concatenate(labels_capa[id_capa][id_grupo]).ravel().tolist()
                    # labels_capa[id_capa][id_grupo] = np.array(labels_capa[id_capa][id_grupo])
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
                puntos_capa[id_capa][id_grupo] = np.array(vector[inicio:fin])
                # siguiente capa para cada grupo
                cont_ptos = cont_ptos + (fin - inicio)  # 03-03-2021
                etiquetas = []
                for i in range((fin - inicio)):
                    etiquetas.append(i)
                labels_capa[id_capa][id_grupo] = np.array(etiquetas)

            inicio = fin


        grupos_capa[id_capa] = npuntos

        # AHORA NO LO NECESITO PORQUE SOLO TENGO UNA CAPA
        # Guardamos los centroides de la capa para poder hacer el proceso inverso
        # vector = puntos_capa[id_capa]
        # for i in range(len(vector)):
        #     vector[i] = np.concatenate(vector[i]).ravel().tolist()  # 03-03-2021
        #     vector[i] = np.array(vector[i])
        #     vector[i] = vector[i].reshape(int(len(vector[i])/2), 2)
        # vector = np.concatenate(vector).ravel().tolist()
        # vector = np.array(vector)
        # vector = vector.reshape(cont_ptos, 2)


        # cant_ptos = cont_ptos  # 03-03-2021 Actualizamos cant_ptos con el número de puntos del siguiente nivel
        # inicio = fin+1

    print("FIN PROCESO CONSTRUCCIÓN")

    end_time_constr = timer()
    logger.info('tree time=%s seconds', end_time_constr - start_time_constr)

    return n_capas, grupos_capa, puntos_capa, labels_capa