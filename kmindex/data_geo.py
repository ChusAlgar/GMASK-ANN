import pandas as pd


def dms_to_dd(d, m, s):
    if d[0]=='-':
        dd = float(d) - float(m)/60 - float(s)/3600
    else:
        dd = float(d) + float(m)/60 + float(s)/3600
    return dd

def processDataGeo(datos_geo):
    longitud = datos_geo['LONGITUD_ETRS89'][:]
    latitud = datos_geo['LATITUD_ETRS89'][:]

    # Transformamos los valores de longitud:
    listlong = [x.split(',') for x in longitud] #longitud.split(',')
    long_grados = []
    cont = 0
    for valores in listlong:
        if (cont == 75):
            print(cont)
        if len(valores) == 4:
            part_ent = valores[2]
            part_dec = valores[3]
            valores[2] = part_ent + '.' + part_dec
        elif len(valores) == 2:
            valores.append('0')
        long_grados.append(dms_to_dd(valores[0], valores[1], valores[2]))
        cont +=1

    # Transformamos los valores de latitud:
    listlat = [x.split(',') for x in latitud]
    lat_grados = []
    for valores in listlat:
        if len(valores) == 4:
            part_ent = valores[2]
            part_dec = valores[3]
            valores[2] = part_ent + '.' + part_dec
        elif len(valores) == 2:
            valores.append('0')
        lat_grados.append(dms_to_dd(valores[0], valores[1], valores[2]))

    long_grados = pd.Series(long_grados)
    lat_grados = pd.Series(lat_grados)
    datos_geo = pd.DataFrame({'LONGITUD':long_grados, 'LATITUD':lat_grados})

    return datos_geo
