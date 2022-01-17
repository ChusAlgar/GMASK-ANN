from pyfaidx import Fasta
import functools
import itertools

secuencias = Fasta('data/SRR5371693_sample.fasta')
sec1 = secuencias[0]
sec1 = sec1[:].seq  # para convertirlo a str
print("sec1: ", sec1)
sec2 = secuencias[1]
sec2 = sec2[:].seq
#print(type(sec2))
print("sec2: ", sec2)
trozo = sec1[:3]
#print(trozo)

def split_seq(seq,size):
    """Divide una secuencia en trozos de tamaño size con solapamiento"""
    return [seq[i:i+size] for i in range(len(seq)-2)]

lista_trozos = split_seq(sec1, 3)
#print(lista_trozos)
#print("num trozos: ", len(lista_trozos))

#count = sum(map(lambda s: s.count(trozo), sec2))
#print(count)

"""add = lambda tr, s2: s2.count(tr)
print(add(sec2, trozo))"""

#names = "SanJoseSanFranciscoSantaFeHouston"
#clave = "San"
#print(type(names))

def f(sec, clav):
    return sec.count(clav)
#count = sum(map(lambda s: s.count(clave), names))
#print([elem for elem in map(f, names, clave)])
#count = sum(map(f, names, clave))
#print(count)

""" No funciona
count2 = sum(map(f, sec2, trozo))
count2 = functools.reduce(lambda x,y: x+y, map(lambda s,clave: s.count(clave), sec2, "CGA"))
print(count2)
print(sec2.count(trozo))"""

"""Prueba para entender como funciona
my_map = map(lambda s,clave: s.count(clave), sec2, ["CGA","CGA"])
print([elem for elem in my_map])"""


"""Prueba para ver como funciona zip
numberList = "ABCDEFGHIJK"
strList = ['one', 'two', 'three']
# Two iterables are passed
for i in range(len(strList)):
    lista.append(numberList)
print(lista)
print(lista.append(numberList) for i in range(len(strList)))
print([[].append(trozo) for trozo in lista_trozos])"""

"""Prueba para buscar un solo trozo
result = itertools.starmap(f, [(sec2, trozo)])
print([elem for elem in result])"""


lista = [sec2[:] for i in range(len(lista_trozos))]
#result = zip(lista, lista_trozos)
#print([elem for elem in result])
result2 = itertools.starmap(f, zip(lista, lista_trozos))
print([elem for elem in result2])