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


def Analyse_Magnitude(datas):

    #Calcule des magnitude
    magnitudes = [round(2/3*math.log(i)-3.2,1) for i in list(datas['e'])]
    Group_magnitudes = Counter(magnitudes)

    # Tri des clef by key, renvoie une liste de tuples 
    lists = sorted(Group_magnitudes.items())   
    x, y = zip(*lists) 

    #Somme des N >= M
    somme=[y[0]]
    for count_mag in reversed(y[1:]) :
        somme = somme + [somme[-1]+count_mag]
    somme.reverse()    

    X = np.asarray(x) # Conversion de tuple vers np.array

    #Calcule de la tangente en un point
    
    x0 = 0
    i0 = np.argmin(np.abs(X-x0))
    x1 = X[i0:i0+2]
    somme1 = somme[i0:i0+2]
    dydx, = np.diff(somme1)/np.diff(x1)
    tngnt = lambda x: dydx*X + (somme1[0]-dydx*x1[0])
    
    #Regression linéaire sur une échelle semilog
    X1 = X.tolist()
    X2 = []
    SOMME2 = []
    for index,i in enumerate(X1) :
        if i > -3.5 and i < 4 :
            X2.append(i)
            SOMME2.append(somme[index])

    X2 = np.asarray(X2)
    p = np.polyfit(X2, np.log(SOMME2), 1)

    #Création et sauvegarde du graphe  
    plt.plot(X, somme)
    plt.plot(x, tngnt(x), label="tangent")
    plt.plot(X2, np.exp(p[0] * X2 + p[1]), 'g--')
    plt.yscale('log')
    plt.savefig('Magnitude_AnalyseTest.png')

    print( "Valeur de b :", p[0])
    print ("Valeur de a :", p[1])


#df=pd.read_csv('../../data/cracks_X1Y2Z01_2k_granite_30MPa_r015.txt', sep= ' ')
df=pd.read_csv('../../data/clustering_results_t_200_v_0point25.csv', sep= ',')   

#df.info()
Analyse_Magnitude(df)