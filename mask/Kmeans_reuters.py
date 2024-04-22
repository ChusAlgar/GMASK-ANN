from random import seed
from random import shuffle

import lectura_reuters as lr
import numpy as np
import utilities as util
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances

# PARÁMETROS DE ENTRADA
#tam = 4  # longitud de las palabras
tam_grupo = 40  # tamaño del grupo (depende de la capacidad computacional).
# Los grupos se cogen por filas completas de la matriz dtm.
#pctg = 0.2  # porcentaje de reducción para calcular el número de centroides de cada etapa
n_centroides = 20 #int((tam_grupo * (4 ** tam) * pctg) / 100)
opcion = 'kmeans'
normaliza = False
# cant_ptos = 200  #200 # número de puntos de cada nube



# DATOS DE ENTRADA: lectura del fichero reuters
tfs, info_cat = lr.read_reuters(1)
tfs_ordenado = tfs.copy()   #guardamos la matriz original

# print("tfs_ordenado")
# for t in tfs_ordenado[0]:
#     print(t)
# Desordenamos los datos:
nfilas, ncolumnas = csr_matrix.get_shape(tfs_ordenado)
# tfs_desordenado = shuffle(tfs_ordenado, random_state=1234)  # NOTA: cuando no tenga que contabilizar los errores de tipo 1,
                                                            # la desordenación la haré con esta función.
seed(1234)
indices = np.arange(tfs_ordenado.shape[0])                  # NOTA: este procedimiento de desordenación lo uso cuando quiero
shuffle(indices)                       # contabilizar los errores de tipo 1 (para poder identificar la categoría)
tfs_desordenado = tfs_ordenado[list(indices)]
# vind = np.linspace(0, nfilas-1, nfilas, dtype=int)
# np.random.shuffle(vind)
# cont = 0
# tfs_desordenado = csr_matrix((nfilas, ncolumnas), dtype=np.int8)     #tfs.copy()
# for elem in vind:
#     tfs_desordenado[cont] = tfs_ordenado[elem]
#     cont += 1



# GRUPOS CAPA 0.
# Calculamos el número de grupos que vamos a formar en la capa 0:
print("CAPA 0. Cálculo de grupos")
ngrupos = int(nfilas / tam_grupo)
if (nfilas % tam_grupo) != 0:
    ngrupos += 1
print("ngrupos: ", ngrupos)

# 17-05-2019. El vector de puntos va a ser un array de arrays donde cada uno de estos será un grupo al que luego se
# aplica el kmeans (o kMedoids)
# vector1 = np.array(np.split(np.array(tfs_desordenado[:(tam_grupo*(ngrupos-1))]), ngrupos-1))
# ult_elem = [np.array(tfs_desordenado[(tam_grupo*(ngrupos-1)):])]
# vector = np.concatenate([vector1, np.array(ult_elem)])
vector = np.array_split(tfs_desordenado.A, ngrupos)
print(len(vector))
# 17-02-2020 dim1v, dim2v, dim3v = vector.shape

print("INICIO PROCESO CONSTRUCCIÓN")
# Inicio del proceso iterativo de construccion-deconstruccion.
iter = 0
continuar = True
error_ant = 0.0
while continuar:
    print('Iteración ', iter)
    # Proceso iterativo para aplicar el kmeans o el kmedoids:
    puntos_capa = []  # estructura en la que almacenamos los centroides de todas las capas
    labels_capa = []  # estructura en la que almacenamos las etiquetas de los puntos
    grupos_capa = []  # estructura en la que almacenamos una lista, para cada capa, que contiene el número de elementos de cada grupo
    #grupos_capa.append(ngrupos)
    id_capa = 0

    indices_resul = []
    while ngrupos >= 1:
        # Capa n:
        #17-05-2019 inicio = 0
        puntos_grupo = []
        labels_grupo = []
        npuntos = []
        print("capa: ", id_capa)
        for id_grupo in range(ngrupos):
            #17-05-2019 fin = inicio + tam_grupo
            #17-05-2019 if fin>len(vector):
            #17-05-2019     fin = len(vector)
            #17-05-2019 npuntos.append(fin-inicio)
            npuntos.append(len(vector[id_grupo]))
            print("grupo: ", id_grupo, "num ptos: ", len(vector[id_grupo]))
            #17-05-2019 if ((fin-inicio)>n_centroides):
            if (len(vector[id_grupo]) > n_centroides):
                if opcion == 'kmeans':
                    #17-05-2019 kmeans = KMeans(n_clusters=n_centroides).fit(vector[inicio:fin])
                    kmeans = KMeans(n_clusters=n_centroides).fit(vector[id_grupo])
                    puntos_grupo.append(kmeans.cluster_centers_)  # aquí tenemos almacenados los puntos de la siguiente capa para cada grupo
                    labels_grupo.append(kmeans.labels_)
                else:
                    #17-05-2019 data = vector[inicio:fin]
                    data = vector[id_grupo]
                    D = pairwise_distances(data, metric='euclidean')
                    M, C = util.kMedoids(D, n_centroides)
                    list_centroides = []
                    for point_idx in M:
                        list_centroides.append(data[point_idx])
                    puntos_grupo.append(np.array(list_centroides))
                    labels_grupo.append(M)
                #17-05-2019 inicio = fin
            else:
                # Si los puntos que tenemos en el grupo no es mayor que el número de centroides, no hacemos culster
                # 17-05-219 puntos_grupo.append(vector[inicio:fin])  # aquí tenemos almacenados los puntos de la siguiente capa
                puntos_grupo.append(vector[id_grupo])  # aquí tenemos almacenados los puntos de la siguiente capa
                                                         # para cada grupo
                etiquetas = []
                # 17-05-2019 for i in range((fin-inicio)):
                for i in range(len(vector[id_grupo])):
                    etiquetas.append(i)
                labels_grupo.append(np.array(etiquetas))
        grupos_capa.append(npuntos)

        # Guardamos los centroides de la capa para poder hacer el proceso inverso
        array = np.array(puntos_grupo)
        puntos_capa.append(array)
        # Guardamos las etiquetas de los puntos (índice del centroide con el que están asociados):
        array2 = np.array(labels_grupo)
        labels_capa.append(array2)
        # Calculamos el numero de grupos de la siguiente capa
        if id_capa==0:
            vector = array.reshape(-1, array.shape[-1])
        else:
            dimension = len(array)  #array.size
            vector = []
            for dim in range(dimension):
                for elem in array[dim]:
                #elemento = array[dim]
                #if len(elemento) == 1:
                #    vector.append(elemento)
                #else:
                #for elem in elemento:
                     vector.append(elem)
            vector = np.array(vector)
        # 18-06-2019 Como el array de arrays puede no tener el mismo número de puntos en todos los grupos, hay que
        # obtener el número de puntos totales de otra manera
        # nfilas, ncolumnas = vector.shape
        # ngrupos = int(nfilas / tam_grupo)
        num_ptos = util.obten_num_ptos(npuntos, n_centroides)
        ngrupos = int(num_ptos / tam_grupo)

        if ngrupos != 0:
            # 18-06-2019 if (nfilas % tam_grupo) != 0:
            if (num_ptos % tam_grupo) != 0:
                ngrupos += 1
            #grupos_capa.append(ngrupos)
            # 17-05-2019 Separamos de nuevo el vector de puntos en array de arrays, donde cada uno de estos es un grupo
            # al que se va aplicar el kmeans (o kMedoids)
            # 18-06-2019 vector = dt.divide_en_grupos(vector, ngrupos, tam_grupo)
            vector = util.divide_en_grupos(vector, num_ptos, ngrupos, tam_grupo)

        id_capa +=1

    print("FIN PROCESO CONSTRUCCIÓN")

#     # Pinto las nubes de puntos originales junto con los centroides sacados
#     #if iter>0:
#     dt.pinta(coordx, coordy, puntos_capa[id_capa-1])
#     # dt.pinta_info_nube(coordx, coordy, puntos_capa, grupos_capa, labels_capa)
#
#     # Pinto los clustters resultado del proceso de construcción:
#     #if iter>0:
#     dt.pinta_clustters(vector_original, labels_capa, puntos_capa)
#
#
    print("********************PROCESO DECONSTRUCCIÓN*********************")
    n_capas = id_capa-1

    lcorrespond = []
    aciertos = 0
    #fallos = 0
    error_index = 0   # el punto no es exáctamente el mismo (el valor de la distancia es distinto de 0) pero la categoría si
    error_clasif = 0   # el punto no es exáctamente el mismo y la categoría tampoco coincide

    #vector_aux = []
    #18-06-2019 vector_encuentra = [ [] for i in range(dim1v) ] #np.zeros((dim1v, dim2v, dim3v))
    nnubes = len(grupos_capa[0])      # 25-02-2020. El número de nubes es el número de grupos de la capa 0
    lista_ptos = [[] for i in range(nnubes)] #donde voy guardando los puntos que busco. A partir de este se construye vector para la siguiente
                     #iteración
    lista_indices = [[] for i in range(nnubes)] # aquí guardo los índices de los puntos para ver como los va ordenando (06-03-2020)

    # 25-02-2020    cont_ptos = np.zeros((dim1v))
    cont_ptos = np.zeros((nnubes))
    for i in range(nfilas):
        seq_buscada = tfs_ordenado[i].toarray()
        #seq_buscada = np.reshape(seq_buscada, (1, 2))

        # f i==20 or i==21:
        #    print("aquí")

        for id_capa in range(n_capas, -1, -1):
            centroides = puntos_capa[id_capa]
            centroides = centroides.reshape(-1, centroides.shape[-1])
            if id_capa < n_capas:
                # seleccionamos solo los puntos que están asociados con ese centroide
                centroidesb = []
                for pos in lista_pos:
                    centroidesb.append(centroides[pos])
                centroides = centroidesb
            puntos_dist = np.concatenate([seq_buscada, centroides])
            D = pairwise_distances(puntos_dist, metric='euclidean')
            columna = util.busca_dist_menor(D)
            # Corrección del índice del centroide
            if id_capa != n_capas:
                pos_centroide = lista_pos[columna - 1]
                if pos_centroide >= n_centroides:
                    id_grupo = int(pos_centroide/n_centroides)
                    id_centroide = pos_centroide - (id_grupo * n_centroides)
                else:
                    id_centroide = pos_centroide
                    id_grupo = 0
                # etiquetas = labels_capa[id_capa + 1]
                # etiquetas = etiquetas.reshape(-1, etiquetas.shape[-1])
                # id_centroide = etiquetas[0][pos_centroide]
            else:
                id_centroide = columna - 1
                id_grupo = 0
            #print(id_centroide)
            lista_pos = []
            #ngrupos, npuntos = labels_capa[id_capa].shape
            ngrupos = len(grupos_capa[id_capa])
            npuntos = grupos_capa[id_capa][id_grupo]
            #for id_grupo in range(ngrupos):
            #fin = labels_capa[id_capa][id_grupo].size
            #for pos in range(fin):
            for pos in range(npuntos):
                if labels_capa[id_capa][id_grupo][pos] == id_centroide:
                    desplaz = 0
                    for id in range(id_grupo):
                        desplaz += grupos_capa[id_capa][id]
                    #lista_pos.append(pos + id_grupo * npuntos)
                    lista_pos.append(pos + desplaz)
            # id_capa=id_capa-1

        # Capa de los datos:
        puntos_seleccionados = []
        for pos in lista_pos:
            # Solo miramos la distancia con los que están en el mismo grupo
            puntos_seleccionados.append(tfs_desordenado[pos].toarray())
        #puntos_dist = np.concatenate([seq_buscada, puntos_seleccionados])
        puntos_dist = [seq_buscada] + puntos_seleccionados
        puntos_dist = np.asarray(puntos_dist)
        puntos_dist = puntos_dist.reshape(-1, puntos_dist.shape[-1])
        D = pairwise_distances(puntos_dist, metric='euclidean')
        columna = util.busca_dist_menor(D)
        id_punto = lista_pos[columna - 1]
        #print("Punto encontrado: ", id_punto,seq_buscada[0],vector_original[id_punto])

        # Contabilizo los errores y aciertos:
        pos = 0
        encontrado = False
        while pos < len(indices) and not encontrado:
            if i == indices[pos]:
                encontrado = True
            else:
                pos += 1
        idgrup_pbuscado = util.obten_idgrupo(pos, grupos_capa[0])
        pos = 0
        encontrado = False
        while pos < len(indices) and not encontrado:
            if id_punto == indices[pos]:
                encontrado = True
            else:
                pos += 1
        idgrup_pencontrado = util.obten_idgrupo(pos, grupos_capa[0])
        if D[0, columna] <= 1.0e-6:
            aciertos += 1
            #Si lo he encontrado, lo guardo en el grupo que le corresponde al punto i
            id_grupo = util.obten_idgrupo(i, grupos_capa[0])
            lista_ptos[id_grupo].append(seq_buscada)
            lista_indices[id_grupo].append(id_punto)
            cont_ptos[id_grupo] += 1
            indices_resul.append(i)

        else:
            cat_pbuscado = lr.busca_categoria(i, info_cat)
            cat_pencontrado = lr.busca_categoria(id_punto, info_cat)
            if cat_pbuscado == cat_pencontrado:
                error_index += 1
            else:
                error_clasif += 1
            # Si no lo he encontrado lo guardo en el grupo en el que está el punto que he encontrado
            # lista_ptos[idgrup_pencontrado].append(tfs_desordenado[id_punto].toarray())
            lista_ptos[idgrup_pbuscado].append(tfs_desordenado[id_punto].toarray())
            # lista_indices[idgrup_pencontrado].append(id_punto)
            lista_indices[idgrup_pbuscado].append(id_punto)
            # cont_ptos[idgrup_pencontrado] += 1
            cont_ptos[idgrup_pbuscado] += 1
            indices_resul.append(id_punto)

        #Metemos el punto que se le asigna en un vector auxiliar (este vector auxiliar será sobre el que volvamos a
        #aplicar el proceso de construcción y deconstrucción:
        # colocamos el punto en la nube que le corresponde:
        #lista_ptos[idgrup_pencontrado].append(seq_buscada)
        #cont_ptos[idgrup_pencontrado] += 1

    # Calculamos el número de grupos que vamos a formar en la capa 0:
    #vector = []
    #for vec in lista_ptos:
    #    vaux = np.array(vec)
    #    vaux = vaux.reshape(-1, vaux.shape[-1])
    #    vector.append(vaux)
        #for elem in vaux:
        #    vector.append(elem)
    #num_ptos = 0
    #for num in cont_ptos:
    #    num_ptos += num
    #ngrupos = int(num_ptos / tam_grupo)
    #if (num_ptos % tam_grupo) != 0:
    #    ngrupos += 1
    #vector = np.array_split(vector, ngrupos)

    # Pasamos la lista de puntos a array de arrays:
    # vector_ptos = []
    # vector_original = []
    # for i in range(len(lista_ptos)):
    #     for elem in lista_ptos[i]:
    #         vector_ptos.append(elem)
    #         vector_original.append((elem[0], elem[1]))
    # vector_ptos = np.array(vector_ptos)
    # # El vector de puntos va a ser un array de arrays donde cada uno de estos será un grupo al que luego se
    # # aplica el kmeans (o kMedoids)
    # vector_ptos = np.array(vector_ptos)
    # vector1 = np.array(np.split(np.array(vector_ptos[:(tam_grupo * (ngrupos - 1))]), ngrupos - 1))
    # ult_elem = [np.array(vector_ptos[(tam_grupo * (ngrupos - 1)):])]
    # # vector1 = vector1.reshape(-1, vector1.shape[-1])
    # vector = np.concatenate([vector1, np.array(ult_elem)])
    # dim1v, dim2v, dim3v = vector.shape
    # ## Hay que corregir la pertenencia a las nubes:
    # # nubes_puntos, puntos_nube, nnubes = util.identifica_nube(tfs_ordenado, tfs_desordenado)

    # Actualizo tfs_desordenado (con respecto a él se calcula la distancia en la capa de los puntos
    # en el proceso de deconstrucción.
    tfs_desordenado = tfs_ordenado[indices_resul]
    vector = np.array_split(tfs_desordenado.A, ngrupos) # parece que funciona mejor si se hace una división más equitativa de los gurpos
    ngrupos = len(vector)

    print("Porcentage de aciertos en la iteración ", iter, ": ", aciertos*100/nfilas)
    # 18-07-2019. Calculamos el error práctio (número ptos erroneos/número ptos totales)
    print("Porcentage de fallos de indexación en la iteración ", iter, ": ", error_index*100/nfilas)
    print("Porcentage de fallos de clasificación en la iteración ", iter, ": ", error_clasif * 100 /nfilas)
    # error = (fallos_t0*100/nfilas) + (fallos_t1 * 100 /nfilas)
    if (error_index + error_clasif)==0 or iter==10:
        continuar = False
    # elif error_index>=(error_ant-error_ant*0.1) and error_index<=(error_ant+error_ant*0.1):
    #    continuar = False
    else:
        iter+=1
        error_ant = error_index



""" Representación del resultado de la deconstrucción"""
'''clustters = []
for i in range (n_centroides):
    clustters.append([puntos_capa[n_capas][0][i]])
for pareja in lcorrespond:
    id_punto, id_clustter = pareja
    clustters[id_clustter].append(vector_original[id_punto])
    #if id_clustter==0:
    #    print("Punto en clustter 0: ", id_punto)

for clustter in clustters:
    print("Tamaño clustter: ", len(clustter))

dt.pinta_result(clustters)'''

print("FIN PROCESO DECONSTRUCCIÓN")