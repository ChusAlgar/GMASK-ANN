import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_data_test():
    np.random.seed(2)
    cant_ptos = 200
    x = np.random.normal(loc=3, scale=1.5, size=cant_ptos)
    y = np.random.normal(loc=3, scale=1.5, size=cant_ptos)
    nube1 = np.array([x+15,y+34])

    x = np.random.normal(loc=2.5, scale=1.5, size=cant_ptos)
    y = np.random.normal(loc=2.5, scale=1.5, size=cant_ptos)
    nube2 = np.array([x+22,y+26])

    x = np.random.normal(loc=2.5, scale=1, size=cant_ptos)
    y = np.random.normal(loc=3, scale=1, size=cant_ptos)
    nube3 = np.array([x+27,y+18])

    x = np.random.normal(loc=2.5, scale=1.5, size=cant_ptos)
    y = np.random.normal(loc=2, scale=1.5, size=cant_ptos)
    nube4 = np.array([x+22,y+10])

    x = np.random.normal(loc=3, scale=1, size=cant_ptos)
    y = np.random.normal(loc=3, scale=1.5, size=cant_ptos)
    nube5 = np.array([x+15,y+2])

    x = np.random.normal(loc=3, scale=1.5, size=cant_ptos)
    y = np.random.normal(loc=2.5, scale=1, size=cant_ptos)
    nube6 = np.array([x+8,y+10])

    x = np.random.normal(loc=2, scale=1.25, size=cant_ptos)
    y = np.random.normal(loc=2, scale=1.25, size=cant_ptos)
    nube7 = np.array([x+3,y+18])

    x = np.random.normal(loc=3.5, scale=1.25, size=cant_ptos)
    y = np.random.normal(loc=2.5, scale=1, size=cant_ptos)
    nube8 = np.array([x+8,y+26])

    return [nube1, nube2, nube3, nube4, nube5, nube6, nube7, nube8]

""" Representación gráfica para ver como son las nubes de puntos
#x = np.arange(0, 30)  # definimos el rango del eje x
nubes = generate_data_test()             # definimos la función a pintar
plt.plot(nubes[0][0,:], nubes[0][1,:], 'go',
        nubes[1][0,:], nubes[1][1,:], 'ro',
        nubes[2][0,:], nubes[2][1,:], 'yo',
        nubes[3][0,:], nubes[3][1,:], 'bo',
        nubes[4][0,:], nubes[4][1,:], 'co',
        nubes[5][0,:], nubes[5][1,:], 'go',
        nubes[6][0,:], nubes[6][1,:], 'mo',
        nubes[7][0,:], nubes[7][1,:], 'ko')
#plt.axis([0, 50, 0, 50])  # se especifica los valores min y max
                         # de los dos ejes [xmin, xmax, ymin, ymax]
                         # Si no acortamos los ejes, se hace automáticamente
plt.title("Representación nubes")
plt.show()"""

def generate_data_test2():
    np.random.seed(2)
    cant_ptos = 200
    x1 = np.random.normal(loc=3, scale=1.5, size=cant_ptos)+15
    y1 = np.random.normal(loc=3, scale=1.5, size=cant_ptos)+34

    x2 = np.random.normal(loc=2.5, scale=1.5, size=cant_ptos)+22
    y2 = np.random.normal(loc=2.5, scale=1.5, size=cant_ptos)+26

    x3 = np.random.normal(loc=2.5, scale=1, size=cant_ptos)+27
    y3 = np.random.normal(loc=3, scale=1, size=cant_ptos)+18

    x4 = np.random.normal(loc=2.5, scale=1.5, size=cant_ptos)+22
    y4 = np.random.normal(loc=2, scale=1.5, size=cant_ptos)+10

    x5 = np.random.normal(loc=3, scale=1, size=cant_ptos)+15
    y5 = np.random.normal(loc=3, scale=1.5, size=cant_ptos)+2

    x6 = np.random.normal(loc=3, scale=1.5, size=cant_ptos)+8
    y6 = np.random.normal(loc=2.5, scale=1, size=cant_ptos)+10

    x7 = np.random.normal(loc=2, scale=1.25, size=cant_ptos)+3
    y7 = np.random.normal(loc=2, scale=1.25, size=cant_ptos)+18

    x8 = np.random.normal(loc=3.5, scale=1.25, size=cant_ptos)+8
    y8 = np.random.normal(loc=2.5, scale=1, size=cant_ptos)+26

    coordx = np.array([x1, x2, x3, x4, x5, x6, x7, x8])
    coordy = np.array([y1, y2, y3, y4, y5, y6, y7, y8])
    coordx = np.reshape(coordx, (1, 8 * cant_ptos))
    coordy = np.reshape(coordy, (1, 8 * cant_ptos))
    #coordx = np.array([x1, x3])
    #coordy = np.array([y1, y3])
    #coordx = np.reshape(coordx, (1, 2 * cant_ptos))
    #coordy = np.reshape(coordy, (1, 2 * cant_ptos))

    return coordx, coordy

"""cx, cy = generate_data_test2()
print('x:', cx.shape)
print('y:', cy.shape)
print(cx)"""

def generate_data_test_overlap():
    np.random.seed(2)
    cant_ptos = 200
    x1 = np.random.normal(loc=3, scale=3.5, size=cant_ptos)+17
    y1 = np.random.normal(loc=3, scale=2.5, size=cant_ptos)+29

    x2 = np.random.normal(loc=2.5, scale=2.5, size=cant_ptos)+26
    y2 = np.random.normal(loc=2.5, scale=2.5, size=cant_ptos)+27

    x3 = np.random.normal(loc=4.5, scale=2.75, size=cant_ptos)+28
    y3 = np.random.normal(loc=4, scale=3.5, size=cant_ptos)+17

    x4 = np.random.normal(loc=4, scale=3, size=cant_ptos)+26
    y4 = np.random.normal(loc=3.5, scale=2.5, size=cant_ptos)+8

    x5 = np.random.normal(loc=3, scale=3, size=cant_ptos)+17
    y5 = np.random.normal(loc=3, scale=3, size=cant_ptos)+4

    x6 = np.random.normal(loc=5, scale=2.5, size=cant_ptos)+4
    y6 = np.random.normal(loc=5.5, scale=3, size=cant_ptos)+6

    x7 = np.random.normal(loc=2, scale=2.25, size=cant_ptos)+3
    y7 = np.random.normal(loc=2, scale=3.75, size=cant_ptos)+18

    x8 = np.random.normal(loc=3.5, scale=2.25, size=cant_ptos)+6
    y8 = np.random.normal(loc=2.5, scale=3.0, size=cant_ptos)+28

    coordx = np.array([x1, x2, x3, x4, x5, x6, x7, x8])
    coordy = np.array([y1, y2, y3, y4, y5, y6, y7, y8])
    coordx = np.reshape(coordx, (1, 8 * cant_ptos))
    coordy = np.reshape(coordy, (1, 8 * cant_ptos))
    #coordx = np.array([x1, x2])
    #coordy = np.array([y1, y2])
    #coordx = np.reshape(coordx, (1, 2 * cant_ptos))
    #coordy = np.reshape(coordy, (1, 2 * cant_ptos))

    return coordx, coordy

#cx, cy = generate_data_test_overlap()
#cx = np.reshape(cx, (8, 200))
#cy = np.reshape(cy, (8, 200))
#colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
#fig, ax = plt.subplots()
#for i in range(8):
#    leyenda = "nube "+ str(i)
#    ax.scatter(cx[i], cy[i], marker='o', color=colors[i], alpha=0.6, label=leyenda)

#ax.legend(bbox_to_anchor=(-0.12, -0.18, 1.2, .10), loc='upper center', ncol=8, mode="expand",
#          borderaxespad=0., fontsize='x-small')
#plt.title("Representación nubes con solape")
#plt.show()

def pinta_vant (centroides):
    nubes = generate_data_test()
    plt.plot(nubes[0][0, :], nubes[0][1, :], 'go',
             nubes[1][0, :], nubes[1][1, :], 'ko',
             nubes[2][0, :], nubes[2][1, :], 'yo',
             nubes[3][0, :], nubes[3][1, :], 'bo',
             nubes[4][0, :], nubes[4][1, :], 'co',
             nubes[5][0, :], nubes[5][1, :], 'go',
             nubes[6][0, :], nubes[6][1, :], 'mo',
             nubes[7][0, :], nubes[7][1, :], 'ko',
             centroides[0][0, 0], centroides[0][0, 1], 'r*',
             centroides[0][1, 0], centroides[0][1, 1], 'r*',
             centroides[0][2, 0], centroides[0][2, 1], 'r*',
             centroides[0][3, 0], centroides[0][3, 1], 'r*',
             centroides[0][4, 0], centroides[0][4, 1], 'r*')
    # plt.axis([0, 50, 0, 50])  # se especifica los valores min y max
    # de los dos ejes [xmin, xmax, ymin, ymax]
    # Si no acortamos los ejes, se hace automáticamente
    plt.title("Representación nubes y centroides")
    plt.show()

def pinta(coordx, coordy, centroides):
    cant_ptos = 200
    coordx = np.reshape(coordx, (8, cant_ptos))
    coordy = np.reshape(coordy, (8, cant_ptos))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    # '#bcbd22', '#17becf', '#1a55FF']

    # Set the plot curve with markers and a title
    fig, ax = plt.subplots()

    for i in range(8):
        leyenda = "nube "+ str(i)
        ax.scatter(coordx[i], coordy[i], marker='o', color=colors[i], label=leyenda)
        #ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    centroides = centroides[0]
    for i in range(len(centroides)):
        ax.scatter(centroides[i,0], centroides[i,1], marker='*', color='black')

    ax.legend(bbox_to_anchor=(-0.12, -0.18, 1.2, .10), loc='upper center', ncol=8, mode="expand",
              borderaxespad=0., fontsize='x-small')
    plt.title("Representación nubes")
    return plt.show()

def pinta_result(clustters):
    nclustters = len(clustters)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    # '#bcbd22', '#17becf', '#1a55FF']

    # Set the plot curve with markers and a title
    fig, ax = plt.subplots()

    cont = 0
    for clustter in (clustters):
        leyenda = "nube "+ str(cont)
        puntos = clustter[1:]
        lpuntos = list(zip(*puntos))
        coordx = lpuntos[0]
        coordy = lpuntos[1]
        ax.scatter(coordx[:], coordy[:], marker='o', color=colors[cont], label=leyenda)
        #ax.scatter(clustter[0][0], clustter[0][1], marker='*', color='black')
        cont +=1

    # Centroides:
    for clustter in (clustters):
        ax.scatter(clustter[0][0], clustter[0][1], marker='*', color='black')

    ax.legend(bbox_to_anchor=(-0.12, -0.18, 1.2, .10), loc='upper center', ncol=8, mode="expand",
              borderaxespad=0., fontsize = 'x-small')
    plt.title("Representación nubes")

    return plt.show()


def pinta_info_nube(coordx, coordy, puntos_capa, grupos_capa, labels_capa):
    n_capas = len(puntos_capa)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    # '#bcbd22', '#17becf', '#1a55FF']

    # Set the plot curve with markers and a title
    fig, ax = plt.subplots()

    #Pintamos la nube 0
    cant_ptos = 200
    coordx = np.reshape(coordx, (n_capas, cant_ptos))
    coordy = np.reshape(coordy, (n_capas, cant_ptos))
    leyenda = "nube 0"
    ax.scatter(coordx[0], coordy[0], marker='o', color='#bcbd22', label=leyenda)

    #Pintamos los centroides de cada capa
    cont = 0
    for id_capa in range(n_capas, 0, -1):
        leyenda = "centroides capa " + str(id_capa-1)

        ngrupos = len(grupos_capa[id_capa-1])
        id_grupo = 0
        npuntos = grupos_capa[id_capa-1][id_grupo]

        puntos = puntos_capa[id_capa-1]
        puntos = puntos.reshape(-1, puntos.shape[-1])
        lista_pos_ant = []
        if id_capa < n_capas:
            # seleccionamos solo los puntos que están asociados con ese centroide
            puntosb = []
            for pos in lista_pos:
                puntosb.append(puntos[pos])
            lpuntos = list(zip(*puntosb))
            coordx = lpuntos[0]
            coordy = lpuntos[1]
            ax.scatter(coordx[:], coordy[:], marker='o', color=colors[cont], label=leyenda)
            lista_pos_ant = lista_pos
        else:
            puntosb = puntos[4]
            #id_centroide = 2
            ax.scatter(puntosb[0], puntosb[1], marker='o', color=colors[cont], label=leyenda)
            lista_pos_ant.append(4)


        lista_pos = []
        for id_centroide in lista_pos_ant:
            for pos in range(npuntos):
                if labels_capa[id_capa-1][id_grupo][pos] == id_centroide:
                    desplaz = 0
                    for id in range(id_grupo):
                        desplaz += grupos_capa[id_capa-1][id]
                    # lista_pos.append(pos + id_grupo * npuntos)
                    lista_pos.append(pos + desplaz)


        cont += 1

    ax.legend(bbox_to_anchor=(-0.12, -0.18, 1.2, .10), loc='upper center', ncol=8, mode="expand",
              borderaxespad=0., fontsize='x-small')
    plt.title("Representación una nube")

    return plt.show()


def pinta_clustters(puntos, labels_capa, puntos_capa):
    ncapas = len(labels_capa)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    markers = ['o','h','s','p','+','v','1','P']
    # Set the plot curve with markers and a title
    fig, ax = plt.subplots()

    centroides = puntos_capa[ncapas - 1]
    centroides = centroides.reshape(-1, centroides.shape[-1])
    ncentroides = len(centroides)
    for idcentroide in range(ncentroides):
        centroide = centroides[idcentroide]
        leyenda = "centroide" + str(idcentroide)
        ax.scatter(centroide[0], centroide[1], marker='*', color=colors[idcentroide], label=leyenda)

        lista_pos = []
        labels = labels_capa[ncapas-1]
        labels = labels[0]
        for idpos in range(len(labels)):
            if idcentroide == labels[idpos]:
                lista_pos.append(idpos)
        for id_capa in range(ncapas-2, -1, -1):
            labels = labels_capa[id_capa]
            encuentra = []
            for pos in range(len(lista_pos)):
                veces = int(lista_pos[pos] // ncentroides)
                elem = lista_pos[pos] - veces*ncentroides
                for idpos in range(len(labels[veces])):
                    if elem == labels[veces][idpos]:
                        desplaz = 0
                        if (lista_pos[pos] >= ncentroides):
                            for i in range(veces):
                                desplaz += len(labels[i])
                        encuentra.append(idpos+desplaz)

            lista_pos = encuentra

        coordx = []
        coordy = []
        for i in lista_pos:
            coordx.append(puntos[i][0])
            coordy.append(puntos[i][1])
        leyenda = "clustter"+str(idcentroide)
        ax.scatter(coordx, coordy, marker=markers[idcentroide], color=colors[idcentroide], label=leyenda)

    ax.legend(bbox_to_anchor=(-0.12, -0.18, 1.2, .10), loc='upper center', ncol=8, mode="expand",
              borderaxespad=0., fontsize='x-small')
    plt.title("Representación una nube")

    return plt.show()
