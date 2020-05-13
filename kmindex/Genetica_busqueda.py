from pyfaidx import Fasta
import functools
import itertools
import shorttext
from sklearn.cluster import KMeans
import numpy as np



def split_seq(seq,size):
    """Divide una secuencia en trozos de tamaño size con solapamiento"""
    return [seq[i:i+size] for i in range(len(seq)-3)]

def seq_count(sec, clav):
    """Cuenta el número de veces que aparece clav en sec"""
    return sec.count(clav)

#Lectura del fichero FASTA y obtencion de las dos primeas secuencias
secuencias = Fasta('data/SRR5371693_sample.fasta')
print(type(secuencias))
''' Esto es una prueba para buscar los trozos de tamaño tam, que se pueden formar en sec1, en sec2:
sec1 = secuencias[0]   # secuencia que vamos a buscar
print(type(sec1))
sec1 = sec1[:].seq     # para convertirlo a str
print("sec1: ", sec1)
sec2 = secuencias[1]
sec2 = sec2[:].seq     # secuencia en la que vamos a buscar
#print(type(sec2))
print("sec2: ", sec2)

#Para el tamaño de palabra tam, buscamos las palabras que se pueden formar en sec1, en sec2
tam = 4
lista_trozos = split_seq(sec1, tam)
lista = [sec2[:] for i in range(len(lista_trozos))]
#result = zip(lista, lista_trozos)
#print([elem for elem in result])
result2 = itertools.starmap(seq_count, zip(lista, lista_trozos))
print([elem for elem in result2])'''

#Lectura del fichero FASTA y almacenamiento de todas las secuencias (documentos)
#Es decir, construimos una lista de documentos
document_list = [secuencias[elem][:].seq for elem in secuencias.keys()]
keys_list = [elem for elem in secuencias.keys()]
#Fijamos el tamaño de las agrupaciones
tam = 4
document_seq_list = [split_seq(secuencia,tam) for secuencia in document_list]
# print(document_seq_list[0])
document_seq_list = [str(' '.join(elem)) for elem in document_seq_list]
#print(type(document_seq_list[0]))
#print(type(document_seq_list))
#print(document_seq_list[0])

#Construimos el corpus (diccionario de palabras), depende del tamaño de palabra que se elija
preprocess = shorttext.utils.standard_text_preprocessor_1()
corpus = [preprocess(document).split(' ') for document in document_seq_list]


''' Prueba para comprobar que funciona la dtm:
result = []
for document in document_seq_list:
    veces = document.count('CCGG')
    print(veces)
    result.append(veces)'''


#Construimos la matriz términos por documento
dtm = shorttext.utils.DocumentTermMatrix(corpus, docids=keys_list)
''' Prueba para comprobar que funciona la dtm:
print(dtm.get_termfreq('SRR5371693.1', 'ccgg'))
ocurrences = dtm.get_token_occurences('ccgg')
print('Ocurrencias de : CCGG', ocurrences)'''


'''
corpus_tam = [len(corpus[i]) for i in range(len(corpus))]
#print('corpus_tam: ', corpus_tam)
lista = [dtm.get_termfreq(keys_list[doc],corpus[doc][term]) for doc, term in range(len(keys_list),len(corpus_tam))]
print(lista[0])
print(lista[1])

lista = []
for term in corpus[0]:
    lista.append(dtm.get_termfreq(keys_list[0],term))
matriz = np.array(lista)
print('1º fila:')
print(len(matriz))
print(matriz)
for i in range(1,len(keys_list)):
    lista = []
    for term in corpus[i]:
        lista.append(dtm.get_termfreq(keys_list[i],term))
    fila = np.array(lista)
    print('fila ', i, ':', len(lista))
    matriz = np.concatenate([matriz, fila])
print('Dimensión: ', matriz.shape)
print('Dimensiones: ', matriz.ndim)
# matriz = np.matrix()
#matriz = matriz.reshape(len(keys_list))
#kmeans = KMeans(n_clusters=10).fit(matriz)
#print(type(kmeans))'''


