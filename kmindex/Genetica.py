from pyfaidx import Fasta

secuencias = Fasta('data/SRR5371693_sample.fasta')
sec1 = secuencias[1]
sec1 = sec1[:].seq  #para convertirlo a str
print("sec1: ", sec1)
#print(len(sec1))
sec2 = secuencias[2]
sec2 = sec2[:].seq
print("sec2: ", sec2)



def trozos(secuencia, tam):
    lista = []
    inicio = 0
    fin = tam
    for i in range(len(secuencia) - 2):
        lista.append(secuencia[inicio:fin])
        inicio += 1
        fin += 1
    return lista


def busca_trozo(secuen2, trozo):
    ''' Busqueda de un trozo en una secuencia'''
    return secuen2.count(trozo)


tam = 3 #tamaño de los trozos que vamos a buscar
lista1_trozos = trozos(sec1, tam)
print(lista1_trozos)
resultado = sum(map(sec2, lista1_trozos))
print(resultado)



# COMPARACIÓN DE DOS SECUENCIAS
'''for i in range(len(secuen1)-2):
    trozo = secuen1[inicio:fin]
    veces = secuen2.count(trozo)
    print("La secuencia ", trozo, " aparece ", veces, " veces")
    inicio += 1
    fin += 1   '''
