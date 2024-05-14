from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from sklearn import preprocessing
import utilities as util
import data_test as dt
from timeit import default_timer as timer




# Parámetros de entada:
#tam = 4  # longitud de las palabras
tam_grupo = 16  # tamaño del grupo (depende de la capacidad computacional).
# Los grupos se cogen por filas completas de la matriz dtm.
#pctg = 0.2  # porcentaje de reducción para calcular el número de centroides de cada etapa
n_centroides = 8 # int((tam_grupo * (4 ** tam) * pctg) / 100)
opcion = 'kmeans'
normaliza = False
cant_ptos = 200 # número de puntos de cada nube

# Datos de entrada: nubes que siguen una distribución normal, en dos dimensiones.
#print("genera los datos")
coordx, coordy = dt.generate_data_test2()  # las genera sin solape
#coordx, coordy = dt.generate_data_test3()  # las genera con un poco de solape
# coordx, coordy = dt.generate_data_test_overlap()
vector_original=list(zip(coordx[0],coordy[0]))
vector_ordenado = list(zip(coordx[0],coordy[0]))
#print("empieza a desordena los datos")
np.random.shuffle(vector_original) #desordenamos los datos
#print("termina de desordena los datos")
#print("empieza la identificación de las nubes")
nubes_puntos, puntos_nube,_ = util.identifica_nube(vector_ordenado, vector_original)
#print("termina la identificación de las nubes")
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
    # Calculamos el número de grupos que vamos a formar en la capa 0:
    nfilas = len(vector)
    ncolumnas = 2
    ngrupos = int(nfilas / tam_grupo)
    if (nfilas % tam_grupo) != 0:
        ngrupos += 1



    # Proceso iterativo para aplicar el kmeans o el kmedoids:
    print("INICIO PROCESO CONSTRUCCIÓN")
    puntos_capa = []  # estructura en la que almacenamos los centroides de todas las capas
    labels_capa = []  # estructura en la que almacenamos las etiquetas de los puntos
    grupos_capa = []  # estructura en la que almacenamos una lista, para cada capa, que contiene el número de elementos de cada grupo
    #grupos_capa.append(ngrupos)
    id_capa = 0
    while ngrupos >= 1:
        # Capa n:
        inicio = 0
        puntos_grupo = []
        labels_grupo = []
        npuntos = []
        for id_grupo in range(ngrupos):
            fin = inicio + tam_grupo
            if fin>len(vector):
                fin = len(vector)
            npuntos.append(fin-inicio)
            if ((fin-inicio)>n_centroides):
                if opcion == 'kmeans':
                    kmeans = KMeans(n_clusters=n_centroides).fit(vector[inicio:fin])
                    puntos_grupo.append(kmeans.cluster_centers_)  # aquí tenemos almacenados los puntos de la siguiente capa para cada grupo
                    labels_grupo.append(kmeans.labels_)
                else:
                    data = vector[inicio:fin]
                    D = pairwise_distances(data, metric='euclidean')
                    M, C = util.kMedoids(D, n_centroides)
                    list_centroides = []
                    for point_idx in M:
                        list_centroides.append(data[point_idx])
                    puntos_grupo.append(np.array(list_centroides))
                    labels_grupo.append(M)
                inicio = fin
            else:
                # Si los puntos que tenemos en el grupo no es mayor que el número de centroides, no hacemos culster
                puntos_grupo.append(vector[inicio:fin])  # aquí tenemos almacenados los puntos de la siguiente capa para cada grupo
                etiquetas = []
                for i in range((fin-inicio)):
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
        vector = array.reshape(-1, array.shape[-1])
        nfilas, ncolumnas = vector.shape
        ngrupos = int(nfilas / tam_grupo)
        #if id_capa == 3:
        #    print("para aquí")
        if ngrupos != 0:
            if (nfilas % tam_grupo) != 0:
                ngrupos += 1
            #grupos_capa.append(ngrupos)
        id_capa +=1

    print("FIN PROCESO CONSTRUCCIÓN")

    end_time_constr = timer()
    print("--- %s seconds ---", end_time_constr-start_time_constr)

    # Pinto las nubes de puntos originales junto con los centroides sacados
    dt.pinta(coordx, coordy, puntos_capa[id_capa-1], 8, 200)
    # dt.pinta_info_nube(coordx, coordy, puntos_capa, grupos_capa, labels_capa)

    start_time_deconstr = timer()

    print("********************PROCESO DECONSTRUCCIÓN*********************")
    n_capas = id_capa-1

    lcorrespond = []
    aciertos = 0
    fallos = 0
    vector_aux = []
    for i in range(len(vector_original)):
        seq_buscada = np.array(vector_original[i])
        seq_buscada = np.reshape(seq_buscada, (1, 2))
        #if i == 6:
        #    print("para aquí")

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
        lcorrespond.append((id_punto, puntos_nube[id_punto])) #id_punto//cant_ptos)) # guardamos el índice del punto encontrado y el índice del clustter al que pertenece
        #Metemos el punto que se le asigna en un vector auxiliar (este vector auxiliar será sobre el que volvamos a aplicar el proceso de
        #construcción y deconstrucción:
        vector_aux.append(vector_original[id_punto])

        #Contamos los aciertos y los fallos
        '''if D[0, 1:].min() == 0.0:
            aciertos+=1
        else:
            fallos+=1'''
        if puntos_nube[i] == puntos_nube[id_punto]:
            aciertos+=1
        else:
            fallos+=1

    vector = vector_aux
    print("Porcentaje de aciertos en la iteración ", iter, ": ", aciertos*100/(cant_ptos*8))
    print("Porcentaje de fallos en la iteración ", iter, ": ", fallos*100/(cant_ptos*8))

end_time_deconstr = timer()
print("--- %s seconds ---", end_time_deconstr-start_time_deconstr)

""" Representación del resultado de la deconstrucción"""
clustters = []
for i in range (n_centroides):
    clustters.append([puntos_capa[n_capas][0][i]])
for pareja in lcorrespond:
    id_punto, id_clustter = pareja
    clustters[id_clustter].append(vector_original[id_punto])
    #if id_clustter==0:
    #    print("Punto en clustter 0: ", id_punto)

for clustter in clustters:
    print("Tamaño clustter: ", len(clustter))

# dt.pinta_result(clustters)

print("FIN PROCESO DECONSTRUCCIÓN")