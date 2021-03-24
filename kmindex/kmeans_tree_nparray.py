from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from sklearn import preprocessing
import utilities as util
import data_test as dt
from timeit import default_timer as timer
import logging

logger = logging.getLogger(__name__)


def kmeans_tree(nclouds, npc, tam_grupo, n_centroides, overlap):
    # Parámetros de entada:
    # tam_grupo = tamaño del grupo para bombardear con los centroides (depende de la capacidad computacional).
    # n_centroides = número de centroides con los que se bombardea cada grupo
    # npc = número de puntos de cada nube
    opcion = 'kmeans'
    normaliza = False

    cant_ptos = nclouds * npc

    # Datos de entrada: nubes que siguen una distribución normal, en dos dimensiones.
    # print("genera los datos")
    if overlap:
        coordx, coordy, indices = dt.generate_data_foverlap(nclouds, npc)  # las genera con un poco de solpae
        # coordx, coordy, indices = dt.generate_data_overlap(nclouds, npc)  #las genera con mucho solape
    else:
        # coordx, coordy = dt.generate_data_test2()  # las genera sin solape
        coordx, coordy, indices = dt.generate_data_noverlap(nclouds, npc)  # las genera sin solape optimizando la
        # pertenencia a la nube

    vector_original = list(zip(coordx[0], coordy[0], indices[0]))
    vector_ordenado = list(zip(coordx[0], coordy[0], indices[0]))

    # print("empieza a desordena los datos")
    np.random.shuffle(vector_original)  # desordenamos los datos

    vector_original, puntos_nube = util.identifica_nube_opt(vector_original)  # optimización de la pertenencia a la nube

    # print("termina de desordenar los datos")
    # print("empieza la identificación de las nubes")
    # nubes_puntos, puntos_nube,_ = util.identifica_nube(vector_ordenado, vector_original)
    # print("termina la identificación de las nubes")
    """cont = 0
    nubes=[]
    nubes_ord=[]
    for i in range(8):
        nube=[]
        for j in range(200):
            nube.append(vector_ordenado[j+200*i])
        nubes.append(nube)
        nubes_ord.append(nube)"""
    """np.random.shuffle(vector_original) #desordenamos los datos
    pos_original = []
    for i in range(8*200):
        for j in range(8*200):
            if vector_original[i] == vector_ordenado[j]:
                pos_original.append((j,i))"""

    # Inicio del proceso iterativo de construcción-deconstrucción.
    start_time_constr = timer()

    vector = vector_original
    for iter in range(1):
        if normaliza:
            vector = preprocessing.normalize(vector, axis=0, norm='l2')

        print("calculo de grupos")
        # Calculamos el número de grupos que vamos a formar en la capa 0 (se obtiene a partir del número de puntos
        # total):
        # Inicio 03-03-2021
        # nfilas = len(vector)
        # ncolumnas = 2
        # ngrupos = int(nfilas / tam_grupo)
        # if (nfilas % tam_grupo) != 0:
        ngrupos = int(cant_ptos / tam_grupo)
        if (cant_ptos % tam_grupo) != 0:
            resto = cant_ptos - (ngrupos * tam_grupo)
            ngrupos = ngrupos + 1
            labels_grupo = np.empty(ngrupos, object)
            alist = [np.zeros((ngrupos - 1, tam_grupo)), np.zeros((1, resto))]
            for i, v in enumerate(alist):
                labels_grupo[i] = v
            if (resto >= n_centroides):
                puntos_grupo = np.zeros((ngrupos, n_centroides, 2), dtype=float)
            else:
                puntos_grupo = np.empty(ngrupos, object)
                alist = [np.zeros((ngrupos - 1, n_centroides, 2)), np.zeros((1, resto, 2))]
                for i, v in enumerate(alist):
                    puntos_grupo[i] = v
        else:
            puntos_grupo = np.zeros((ngrupos, n_centroides, 2), dtype=float)
            labels_grupo = np.zeros((ngrupos, tam_grupo), dtype=int)
        # Fin 03-03-2021

        # Proceso iterativo para aplicar el kmeans o el kmedoids:
        print("INICIO PROCESO CONSTRUCCIÓN")
        puntos_capa = []  # estructura en la que almacenamos los centroides de todas las capas
        labels_capa = []  # estructura en la que almacenamos las etiquetas de los puntos
        grupos_capa = []  # estructura en la que almacenamos una lista, para cada capa, que contiene el número de
        # elementos de cada grupo
        # grupos_capa.append(ngrupos)
        id_capa = 0
        while ngrupos >= 1:
            # Capa n:
            inicio = 0
            # 18-03-2021 puntos_grupo y labels_grupo ahora van a ser un np.array de tres dimensiones y los calculo
            # cuando calculo el número de grupos
            # puntos_grupo = []
            # labels_grupo = []

            cont_ptos = 0  # 03-03-2021. Contador de los puntos en cada capa
            npuntos = []
            for id_grupo in range(ngrupos):
                fin = inicio + tam_grupo
                # Inicio 03-03-2021. Control del último grupo (no tiene cantidad de puntos suficientes para formar
                # grupo
                # if fin > len(vector):
                if fin > cant_ptos:
                    # fin = len(vector)
                    fin = cant_ptos
                # Fin 03-03-2021
                npuntos.append(fin - inicio)
                if ((fin - inicio) >= n_centroides):
                    if opcion == 'kmeans':
                        kmeans = KMeans(n_clusters=n_centroides).fit(vector[inicio:fin])
                        # 18-03-2021 puntos_grupo y labels_grupo ahora van a ser un np.array de tres dimensiones
                        # puntos_grupo.append(kmeans.cluster_centers_)  # aquí tenemos almacenados los puntos de la
                        # siguiente capa para cada grupo
                        # labels_grupo.append(kmeans.labels_)
                        puntos_grupo[id_grupo] = kmeans.cluster_centers_
                        labels_grupo[id_grupo] = kmeans.labels_
                        cont_ptos += n_centroides  # 03-03-20021

                    else:
                        data = vector[inicio:fin]
                        D = pairwise_distances(data, metric='euclidean')
                        M, C = util.kMedoids(D, n_centroides)
                        list_centroides = []
                        for point_idx in M:
                            list_centroides.append(data[point_idx])
                        puntos_grupo.append(np.array(list_centroides))
                        cont_ptos += n_centroides  # 03-03-20021
                        labels_grupo.append(M)
                    inicio = fin
                else:
                    # Si los puntos que tenemos en el grupo no es mayor que el número de centroides, no hacemos culster
                    # 03-03-2021  puntos_grupo.append(vector[inicio:fin])  # aquí tenemos almacenados los puntos de la
                    # siguiente capa para cada grupo
                    # 18-03-2021 puntos_grupo.append(np.array(vector[inicio:fin]))  # aquí tenemos almacenados los puntos de la
                    puntos_grupo[id_grupo] = np.array(vector[inicio:fin])
                    # siguiente capa para cada grupo
                    cont_ptos = cont_ptos + (fin - inicio)  # 03-03-2021
                    etiquetas = []
                    for i in range((fin - inicio)):
                        etiquetas.append(i)
                    # 18-03-2021 labels_grupo.append(np.array(etiquetas))
                    labels_grupo[id_grupo] = np.array(etiquetas)
            grupos_capa.append(npuntos)

            # Guardamos los centroides de la capa para poder hacer el proceso inverso
            # 18-03-2021 array = np.array(puntos_grupo)
            # 18-03-2021 puntos_capa.append(array)
            puntos_capa.append(puntos_grupo)
            # Guardamos las etiquetas de los puntos (índice del centroide con el que están asociados):
            # 18-03-2021 array2 = np.array(labels_grupo)
            # 18-03-2021 labels_capa.append(array2)
            labels_capa.append(labels_grupo)
            # Almacenamos en vector los puntos de la siguiente capa
            # 03-03-2021 vector = array.reshape(-1, array.shape[-1])
            # 18-03-2021 vector = np.concatenate(puntos_grupo).ravel().tolist()  # 03-03-2021
            # 18-03-2021 vector = np.array(vector)
            # 18-03-2021 vector = vector.reshape(cont_ptos, 2)

            # vector = puntos_grupo
            # vector = vector.reshape(-1, vector.shape[-1])
            vector = np.concatenate(puntos_grupo).ravel().tolist()  # 03-03-2021
            vector = np.array(vector)
            vector = vector.reshape(cont_ptos, 2)

            # Calculamos el numero de grupos de la siguiente capa
            ngrupos = int(cont_ptos / tam_grupo)  # 03-03-2021  nfilas, ncolumnas = vector.shape
            if ngrupos != 0:
                if (cont_ptos % tam_grupo) != 0:  # 03-03-2021 if (nfilas % tam_grupo) != 0:
                    resto = cont_ptos - (ngrupos * tam_grupo)
                    ngrupos = ngrupos + 1
                    labels_grupo = np.empty(ngrupos, object)
                    alist = [np.zeros((ngrupos-1,tam_grupo)), np.zeros((1,resto))]
                    for i, v in enumerate(alist):
                        labels_grupo[i] = v
                    if (resto >= n_centroides):
                        puntos_grupo = np.zeros((ngrupos, n_centroides, 2), dtype=float)
                    else:
                        puntos_grupo = np.empty(ngrupos, object)
                        alist = [np.zeros((ngrupos - 1, n_centroides,2)), np.zeros((1,resto,2))]
                        for i, v in enumerate(alist):
                            puntos_grupo[i] = v
                else:
                    puntos_grupo = np.zeros((ngrupos, n_centroides, 2), dtype=float)
                    labels_grupo = np.zeros((ngrupos, tam_grupo), dtype=int)
                # grupos_capa.append(ngrupos)
            cant_ptos = cont_ptos  # 03-03-2021 Actualizamos cant_ptos con el número de puntos del siguiente nivel
            id_capa += 1

        print("FIN PROCESO CONSTRUCCIÓN")

        end_time_constr = timer()
        # print("--- %s seconds ---", end_time_constr-start_time_constr)
        logger.info('tree time=%s seconds', end_time_constr - start_time_constr)

        # Pinto las nubes de puntos originales junto con los centroides sacados
        dt.pinta(coordx, coordy, puntos_capa[id_capa - 1], npc, nclouds)
        # dt.pinta_info_nube(coordx, coordy, puntos_capa, grupos_capa, labels_capa)

        start_time_deconstr = timer()

        print("********************PROCESO DECONSTRUCCIÓN*********************")
        n_capas = id_capa - 1
        logger.info('tree-depth=%s', n_capas)

        lcorrespond = []
        aciertos = 0
        fallos = 0
        vector_aux = []
        for i in range(len(vector_original)):
            # print('buscando punto ', i)
            seq_buscada = np.array(vector_original[i])
            seq_buscada = np.reshape(seq_buscada, (1, 2))
            # seq_buscada = np.reshape(seq_buscada, (1, 3))
            # indice = seq_buscada[0,2]
            # coordx = seq_buscada[0,0]
            # coordy = seq_buscada[0,1]
            # seq_buscada = np.array([coordx, coordy])

            for id_capa in range(n_capas, -1, -1):
                # 18-03-2021 Ahora puntos_capa es una list de ndarrays
                # 03-03-2021 Obtenemos los centroides de la capa

                centroides = puntos_capa[id_capa]
                # 18-03-2021 centroides = np.array(puntos_capa[id_capa])
                # centroides = centroides.reshape(-1, centroides.shape[-1])
                # centroides = np.concatenate(centroides).ravel().tolist()
                centroides = np.concatenate(centroides) #.ravel() #.tolist()
                # dim = int(len(centroides) / 2)
                # dim1, dim2, _ = centroides.shape
                # dim = dim1*dim2
                # centroides = np.array(centroides)
                # centroides = centroides.reshape(dim, 2)
                # Fin 18-03-2021

                if id_capa < n_capas:
                    # seleccionamos solo los puntos que están asociados con ese centroide
                    centroidesb = np.zeros((len(lista_pos),2), dtype=float)
                    cont = 0
                    for pos in lista_pos:
                        # centroidesb.append(centroides[pos])
                        centroidesb[cont] = centroides[pos]
                        cont += 1
                    centroides = centroidesb

                puntos_dist = np.concatenate([seq_buscada, centroides])
                D = pairwise_distances(puntos_dist, metric='euclidean')
                columna = util.busca_dist_menor(D)
                # Corrección del índice del centroide
                if id_capa != n_capas:
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
                    if len(grupos_capa[n_capas]) > 1:
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
                lista_pos = []
                # ngrupos, npuntos = labels_capa[id_capa].shape
                ngrupos = len(grupos_capa[id_capa])
                npuntos = grupos_capa[id_capa][id_grupo]
                # for id_grupo in range(ngrupos):
                # fin = labels_capa[id_capa][id_grupo].size
                # for pos in range(fin):
                for pos in range(npuntos):
                    if labels_capa[id_capa][id_grupo][pos] == id_centroide:
                        desplaz = 0
                        for id in range(id_grupo):
                            desplaz += grupos_capa[id_capa][id]
                        # lista_pos.append(pos + id_grupo * npuntos)
                        lista_pos.append(pos + desplaz)
                # id_capa=id_capa-1

            # Capa de los datos:
            puntos_seleccionados = []
            for pos in lista_pos:
                # Solo miramos la distancia con los que están en el mismo grupo
                puntos_seleccionados.append(vector_original[pos])
            puntos_dist = np.concatenate([seq_buscada, puntos_seleccionados])
            D = pairwise_distances(puntos_dist, metric='euclidean')
            columna = util.busca_dist_menor(D)
            id_punto = lista_pos[columna - 1]
            # print("Punto encontrado: ", id_punto,seq_buscada[0],vector_original[id_punto])
            # guardamos el índice del punto encontrado y el índice del clustter al que pertenece
            # 17-02-2021  lcorrespond.append((id_punto, puntos_nube[id_punto])) #id_punto//cant_ptos))
            # Metemos el punto que se le asigna en un vector auxiliar (este vector auxiliar será sobre el que volvamos a
            # aplicar el proceso de construcción y deconstrucción:
            vector_aux.append(vector_original[id_punto])

            # Contamos los aciertos y los fallos
            '''if D[0, 1:].min() == 0.0:
                aciertos+=1
            else:
                fallos+=1'''
            if puntos_nube[i] == puntos_nube[id_punto]:
                aciertos += 1
            else:
                fallos += 1

        vector = vector_aux
        print("Porcentaje de aciertos en la iteración ", iter, ": ", aciertos * 100 / (npc * nclouds))
        print("Porcentaje de fallos en la iteración ", iter, ": ", fallos * 100 / (npc * nclouds))
        logger.info('Porcentaje de aciertos= %s', aciertos * 100 / (npc * nclouds))
        logger.info('Porcentaje de fallos= %s', fallos * 100 / (npc * nclouds))

    end_time_deconstr = timer()
    # print("--- %s seconds ---", end_time_deconstr-start_time_deconstr)
    logger.info('search time= %s seconds', end_time_deconstr - start_time_deconstr)

    """ Representación del resultado de la deconstrucción"""
    '''17-02-2021   clustters = []
    for i in range (n_centroides):
        clustters.append([puntos_capa[n_capas][0][i]])
    for pareja in lcorrespond:
        id_punto, id_clustter = pareja
        clustters[id_clustter].append(vector_original[id_punto])
        #if id_clustter==0:
        #    print("Punto en clustter 0: ", id_punto)

    #for clustter in clustters:
    #    print("Tamaño clustter: ", len(clustter))

    dt.pinta_result(clustters)'''  # 17-02-2021

    print("FIN PROCESO DECONSTRUCCIÓN")