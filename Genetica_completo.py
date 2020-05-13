from pyfaidx import Fasta
import shorttext
from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from sklearn import preprocessing
import utilities as util




# Parámetros de entada:
tam = 4  # longitud de las palabras
tam_grupo = 10  # tamaño del grupo (depende de la capacidad computacional).
# Los grupos se cogen por filas completas de la matriz dtm.
pctg = 0.2  # porcentaje de reducción para calcular el número de centroides de cada etapa
n_centroides = int((tam_grupo * (4 ** tam) * pctg) / 100)
opcion = 'kmeans'
normaliza = False

# Lectura del fichero FASTA y almacenamiento de todas las secuencias (documentos)
secuencias = Fasta('data/SRR5371693_sample.fasta')
print("fichero leido")
# Es decir, construimos una lista de documentos
document_list = [secuencias[elem][:].seq for elem in secuencias.keys()]
keys_list = [elem for elem in secuencias.keys()]
document_seq_list = [util.split_seq(secuencia, tam) for secuencia in document_list]
document_seq_list = [str(' '.join(elem)) for elem in document_seq_list]

# Construimos el corpus (diccionario de palabras), depende del tamaño de palabra que se elija
print("construcción del corpus")
preprocess = shorttext.utils.standard_text_preprocessor_1()
corpus = [preprocess(document).split(' ') for document in document_seq_list]

print("construcción dtm")
# Construimos la matriz términos por documento
dtm = shorttext.utils.DocumentTermMatrix(corpus, docids=keys_list)
# ''' Prueba para comprobar que funciona la dtm:'''
# print(dtm.get_termfreq('SRR5371693.1', 'ccgg'))
# ocurrences = dtm.get_token_occurences('ccgg')
# print('Ocurrencias de : CCGG', ocurrences)
matriz = dtm.dtm.todense()
vector_original = np.asarray(matriz, dtype=np.int)
vector = vector_original
if normaliza:
    vector = preprocessing.normalize(vector, axis=0, norm='l2')

print("calculo de grupos")
# Calculamos el número de grupos que vamos a formar en la capa 0:
nfilas, ncolumnas = vector.shape
ngrupos = int(nfilas / tam_grupo)
if (nfilas % tam_grupo) != 0:
    ngrupos += 1



# Proceso iterativo para aplicar el kmeans o el kmedoids:
print("INICIO PROCESO CONSTRUCCIÓN")
puntos_capa = []  # estructura en la que almacenamos los centroides de todas las capas
labels_capa = []  # estructura en la que almacenamos las etiquetas de los puntos
grupos_capa = []  # estructura en la que almacenamos una lista, para cada capa, que contiene el número de elementos de cada grupo
#grupos_capa.append(ngrupos)
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
    if ngrupos != 0:
        if (nfilas % tam_grupo) != 0:
            ngrupos += 1
        #grupos_capa.append(ngrupos)

print("FIN PROCESO CONSTRUCCIÓN")

print("********************PROCESO DECONSTRUCCIÓN*********************")
seq_buscada = vector_original[16]
seq_buscada.resize(1, 4**tam)
n_capas = len(puntos_capa)

for id_capa in range(n_capas - 1, -1, -1):
    centroides = puntos_capa[id_capa]
    centroides = centroides.reshape(-1, centroides.shape[-1])
    if id_capa < n_capas - 1:
        # seleccionamos solo los puntos que están asociados con ese centroide
        centroidesb = []
        for pos in lista_pos:
            centroidesb.append(centroides[pos])
        centroides = centroidesb
    puntos_dist = np.concatenate([seq_buscada, centroides])
    D = pairwise_distances(puntos_dist, metric='euclidean')
    columna = util.busca_dist_menor(D)
    # Corrección del índice del centroide
    if id_capa != n_capas - 1:
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
print("Punto encontrado: ", id_punto)

print("FIN PROCESO DECONSTRUCCIÓN")
