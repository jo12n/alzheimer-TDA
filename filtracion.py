import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib import cm
import tadasets
import persim

def barcode2(serie):
    tam_serie = len(serie)
    ind_ord = np.argsort(serie)
    m = ind_ord[0]
    barcode = [[serie[m],serie[m]]]
    ind_bar = {m:0}     
    for i in ind_ord[1:]:
        ind_car = []
        ven_izq = i-1
        ven_der= i+1
        while ven_izq >= 0 and serie[ven_izq] < serie[i]:
            ven_izq = ven_izq-1
        if ven_izq < 0:
            ven_izq = 0
        while ven_der < tam_serie and serie[ven_der] < serie[i]:
            ven_der = ven_der +1
        if ven_der >= tam_serie-1:
            ven_der = tam_serie
        for intervalo in list(set(range(ven_izq,ven_der)).intersection(set(ind_bar.keys()))):
            if max(serie[min([i,intervalo]):max([i,intervalo])]) <= serie[i]:
                ind_car.append(intervalo)
        if len(ind_car) == 0:
            barcode.append([serie[i],serie[i]])
            ind_bar.update({i:len(ind_bar)})

        elif len(ind_car) == 1:
            continue
        else:
            p = min([serie[ind_bar[w]] for w in ind_car])
            for w in ind_car:
                if barcode[ind_bar[w]][0] == barcode[ind_bar[w]][1] and serie[ind_bar[w]] > p:
                    barcode[ind_bar[w]][1] = serie[i]
    barcode[0][1] = max(serie)
    return np.array(barcode)
