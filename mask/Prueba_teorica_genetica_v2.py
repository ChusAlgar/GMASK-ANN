from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from sklearn import preprocessing
import utilities as util
import data_test as dt




# Parámetros de entada:
#tam = 4  # longitud de las palabras
tam_grupo = 16  # tamaño del grupo (depende de la capacidad computacional).
# Los grupos se cogen por filas completas de la matriz dtm.
#pctg = 0.2  # porcentaje de reducción para calcular el número de centroides de cada etapa
n_centroides = 8 #int((tam_grupo * (4 ** tam) * pctg) / 100)
opcion = 'kmeans'
normaliza = False
cant_ptos = 200  #200 # número de puntos de cada nube

# Datos de entrada: nubes que siguen una distribución normal, en dos dimensiones.
# coordx, coordy = dt.generate_data_test2()
coordx, coordy = dt.generate_data_test_overlap()
vector_original=list(zip(coordx[0],coordy[0]))
vector_ordenado = list(zip(coordx[0],coordy[0]))
np.random.shuffle(vector_original) #desordenamos los datos
nubes_puntos, puntos_nube, nnubes = util.identifica_nube(vector_ordenado, vector_original)
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



# 17-05-2019 vector = vector_original
if normaliza:
    # 17-05-2019 vector = preprocessing.normalize(vector, axis=0, norm='l2')
    vector_original = preprocessing.normalize(vector_original, axis=0, norm='l2')

print("calculo de grupos")
# Calculamos el número de grupos que vamos a formar en la capa 0:
nfilas = len(vector_original)
ncolumnas = 2
ngrupos = int(nfilas / tam_grupo)
if (nfilas % tam_grupo) != 0:
    ngrupos += 1
# 17-05-2019. El vector de puntos va a ser un array de arrays donde cada uno de estos será un grupo al que luego se
# aplica el kmeans (o kMedoids)
vector1 = np.array(np.split(np.array(vector_original[:(tam_grupo*(ngrupos-1))]), ngrupos-1))
ult_elem = [np.array(vector_original[(tam_grupo*(ngrupos-1)):])]
#vector1 = vector1.reshape(-1, vector1.shape[-1])
vector = np.concatenate([vector1, np.array(ult_elem)])
dim1v, dim2v, dim3v = vector.shape


# Inicio del proceso iterativo de consutrccuín-deconstrucción.
iter = 0
continuar = True
error_ant = 0.0
while continuar:
    print('Iteración ', iter)
    # Proceso iterativo para aplicar el kmeans o el kmedoids:
    print("INICIO PROCESO CONSTRUCCIÓN")
    puntos_capa = []  # estructura en la que almacenamos los centroides de todas las capas
    labels_capa = []  # estructura en la que almacenamos las etiquetas de los puntos
    grupos_capa = []  # estructura en la que almacenamos una lista, para cada capa, que contiene el número de elementos de cada grupo
    #grupos_capa.append(ngrupos)
    id_capa = 0
    while ngrupos >= 1:
        # Capa n:
        #17-05-2019 inicio = 0
        puntos_grupo = []
        labels_grupo = []
        npuntos = []
        for id_grupo in range(ngrupos):
            #17-05-2019 fin = inicio + tam_grupo
            #17-05-2019 if fin>len(vector):
            #17-05-2019     fin = len(vector)
            #17-05-2019 npuntos.append(fin-inicio)
            npuntos.append(len(vector[id_grupo]))
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

    # Pinto las nubes de puntos originales junto con los centroides sacados
    #if iter>0:
    dt.pinta(coordx, coordy, puntos_capa[id_capa-1], 8, 200)
    # dt.pinta_info_nube(coordx, coordy, puntos_capa, grupos_capa, labels_capa)

    # Pinto los clustters resultado del proceso de construcción:
    #if iter>0:
    dt.pinta_clustters(vector_original, labels_capa, puntos_capa)


    print("********************PROCESO DECONSTRUCCIÓN*********************")
    n_capas = id_capa-1

    lcorrespond = []
    aciertos = 0
    fallos = 0
    #vector_aux = []
    #18-06-2019 vector_encuentra = [ [] for i in range(dim1v) ] #np.zeros((dim1v, dim2v, dim3v))
    lista_ptos = [[] for i in range(nnubes)] #donde voy guardando los puntos que busco. A partir de este se construye vector para la siguiente
                     #iteración
    cont_ptos = np.zeros((dim1v))
    for i in range(len(vector_original)):
        seq_buscada = np.array(vector_original[i])
        seq_buscada = np.reshape(seq_buscada, (1, 2))

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
            puntos_seleccionados.append(vector_original[pos])
        puntos_dist = np.concatenate([seq_buscada, puntos_seleccionados])
        D = pairwise_distances(puntos_dist, metric='euclidean')
        columna = util.busca_dist_menor(D)
        id_punto = lista_pos[columna - 1]
        #print("Punto encontrado: ", id_punto,seq_buscada[0],vector_original[id_punto])

        #Metemos el punto que se le asigna en un vector auxiliar (este vector auxiliar será sobre el que volvamos a
        #aplicar el proceso de construcción y deconstrucción:
        #17-05-2019 vector_aux.append(vector_original[id_punto])
        # vector_encuentra[id_punto//dim2v, int(cont_ptos[id_punto//dim2v]), :] = np.array(vector_original[id_punto])
        id_grupo = util.obten_idgrupo(id_punto, grupos_capa[0])
        # guardamos el índice del punto encontrado y el índice del clustter al que pertenece
        lcorrespond.append((id_punto, puntos_nube[id_punto]))   #((id_punto, id_grupo))
        #Contamos los aciertos y los fallos
        '''if D[0, 1:].min() == 0.0:
            aciertos+=1
            vector_encuentra[id_grupo].append(np.array(vector_original[id_punto]))
        else:
            fallos+=1
            vector_encuentra[id_grupo].append(np.array(vector_original[i]))'''
        if puntos_nube[i] == puntos_nube[id_punto]:
            aciertos+=1
        else:
            fallos+=1
        # colocamos el punto en la nube que le corresponde:
        id_nube = puntos_nube[id_punto]
        lista_ptos[id_nube].append(np.array(vector_original[i]))
        cont_ptos[id_grupo] += 1

    # 17-05-2019 vector = vector_aux
    # Lo pasamos a np.array
    #18-06-2019 vector = [np.array(vector_encuentra[i]) for i in range(dim1v)]
    # Calculamos el número de grupos que vamos a formar en la capa 0:
    num_ptos = 0
    for num in cont_ptos:
        num_ptos += num
    ngrupos = int(num_ptos / tam_grupo)
    if (num_ptos % tam_grupo) != 0:
        ngrupos += 1
    # Pasamos la lista de puntos a array de arrays:
    vector_ptos = []
    vector_original = []
    for i in range(len(lista_ptos)):
        for elem in lista_ptos[i]:
            vector_ptos.append(elem)
            vector_original.append((elem[0], elem[1]))
    vector_ptos = np.array(vector_ptos)
    # El vector de puntos va a ser un array de arrays donde cada uno de estos será un grupo al que luego se
    # aplica el kmeans (o kMedoids)
    vector_ptos = np.array(vector_ptos)
    vector1 = np.array(np.split(np.array(vector_ptos[:(tam_grupo * (ngrupos - 1))]), ngrupos - 1))
    ult_elem = [np.array(vector_ptos[(tam_grupo * (ngrupos - 1)):])]
    # vector1 = vector1.reshape(-1, vector1.shape[-1])
    vector = np.concatenate([vector1, np.array(ult_elem)])
    dim1v, dim2v, dim3v = vector.shape
    # Hay que corregir la pertenencia a las nubes:
    nubes_puntos, puntos_nube, nnubes = util.identifica_nube(vector_ordenado, vector_original)

    #print([elem for elem in lcorrespond])
    print("Porcentaje de aciertos en la iteración ", iter, ": ", aciertos*100/(cant_ptos*8))
    # 18-07-2019. Calculamos el error práctio (número ptos erroneos/número ptos totales)
    error = fallos*100/(8 * cant_ptos)
    print("Porcentaje de fallos (error) en la iteración ", iter, ": ", error)
    if fallos==0 or iter==10:
        continuar = False
    elif error>=(error_ant-error_ant*0.01) and error<=(error_ant+error_ant*0.01):
        continuar = False
    else:
        iter+=1
        error_ant = error



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