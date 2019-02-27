'''
Date 08/02/2019 

Algorithme d'analyse en magnitude de données géologiques.
'''

from collections import Counter
from scipy import interpolate

import matplotlib

matplotlib.use("svg")
import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np


def analyse_magnitude(datas):

    plt.figure()
    # Calcule des magnitude
    magnitudes = [round(2 / 3 * math.log(i) - 3.2, 2) for i in list(datas['e'])]
    group_magnitudes = Counter(magnitudes)

    # Tri des clef by key, renvoie une liste de tuples 
    lists = sorted(group_magnitudes.items())
    x, y = zip(*lists)

    # Somme des N >= M
    somme = [y[0]]
    for count_mag in reversed(y[1:]):
        somme = somme + [somme[-1] + count_mag]
    somme.reverse()

    X = np.asarray(x)  # Conversion de tuple vers np.array
    SOMME = np.asarray(somme)

    # Calcule de la tangente en un point

    # x0 = 0
    # i0 = np.argmin(np.abs(X-x0))
    # x1 = X[i0:i0+2]
    # somme1 = somme[i0:i0+2]
    # dydx, = np.diff(somme1)/np.diff(x1)
    # tngnt = lambda x: dydx*X + (somme1[0]-dydx*x1[0])

    # Regression linéaire sur une échelle semilog
    (best_min,best_max) = recherche_borne(X,SOMME, 3 )
    condition = [i > best_min and i < best_max for i in X]
    X2 = X[condition]
    SOMME2 = SOMME[condition]
    p = np.polyfit(X2, np.log(SOMME2), 1)

    
    # Création et sauvegarde du graphe
    plt.plot(X, somme)
    # plt.plot(x, tngnt(x), label="tangent")
    plt.plot(X2, np.exp(p[0] * X2 + p[1]), 'g--')
    plt.yscale('log')
    plt.savefig('../data/Magnitude_AnalyseTest.png')

    print("Valeur de b :", p[0])
    print("Valeur de a :", p[1])

    return {
        "a": p[1],
        "b": p[0]
    }

def residus (a, b, magnitudes, somme):
    somme_residus = 0
    somme_cumul = 0    

    for index, mag in enumerate(magnitudes):
        predicted_cumul = (a+b*mag)
        # print('predicted_cumul = ', predicted_cumul)
        somme_residus += abs(somme[index]- predicted_cumul)
        somme_cumul += somme[index]

    residus_f = 100 - 100*somme_residus/somme_cumul
    #print('somme résidus = ', somme_residus)
    #print('somme_cumul = ', somme_cumul)
    #print('résidus = ', residus_f )       
    return residus_f

def recherche_borne(magnitudes, somme, intervalle_min = 3):
    best_min = float('-inf')
    best_max = float('inf')
    index_min = 1
    residus_limite = 1.5
    n = len(magnitudes)
    index_max = n

    i = 1
    residus_actuel = 0
    while i < n  and residus_actuel < residus_limite :  
        print('BOUCLE RECHER MIN')
        magnitude_min = magnitudes[i]
        condition = [i <= magnitude_min for i in magnitudes]
        mag = magnitudes[condition]
        sum1 = somme[condition]
        p = np.polyfit(mag, np.log(sum1), 1)
        residus_actuel = residus(p[1],p[0], mag, sum1)
        plt.plot(mag, np.exp(p[0] * mag + p[1]))
        print(residus_actuel, magnitude_min, p[1], p[0])
        if residus_actuel >= residus_limite :
            best_min = magnitude_min
            index_min = i
        i += 1

    i = n-2
    residus_actuel = 0
    while i >= 0 and residus_actuel < residus_limite :  
        print('BOUCLE RECHER MAX')
        magnitude_max = magnitudes[i]
        condition = [i >= magnitude_max for i in magnitudes]
        mag = magnitudes[condition]
        sum1 = somme[condition]
        p = np.polyfit(mag, np.log(sum1), 1)
        residus_actuel = residus( p[1], p[0], mag, sum1)
        plt.plot(mag, np.exp(p[0] * mag + p[1]))
        print(residus_actuel, magnitude_max, p[1], p[0])
        if residus_actuel >= residus_limite :
                best_max = magnitude_max
                index_max = i
        i -= 1

    if best_max <= best_min : 
        best_max = float('inf')
    if index_max-index_min <= intervalle_min : 
        print('WARNING INTERVALLE DE REGRESSION TROUVE TROP FAIBLE')
     
    print(best_min, index_min)
    print(best_max, index_max)
    return (best_min, best_max)


        
    for index_max, magnitude_max in reversed(list(enumerate(magnitudes))):
            #LE PRETRATEMENT SVP

            residus_actuel = residus(a,b, magnitudes, somme)
            if residus_actuel < best_residus :
                best_min = magnitude_min
                best_max = magnitude_max
                best_residus = residus_actuel
    
    return 




if __name__ == "__main__":
    #df=pd.read_csv('../data/crack.csv', sep= ' ')
    df = pd.read_csv('../data/clustering_results_t_200_v_0point25.csv', sep=',')
    # df.info()

 

    analyse_magnitude(df)

    