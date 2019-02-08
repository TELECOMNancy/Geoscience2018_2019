'''
Date 08/02/2019 

Algorithme d'analyse en magnitude de données géologiques.
'''

from collections import Counter

import matplotlib
matplotlib.use("svg")
import pandas as pd 
import matplotlib.pyplot as plt
import math

def Analyse_Magnitude(datas):
    magnitudes = [round(2/3*math.log(i)-3.2,1) for i in list(datas['e'])]
    Group_magnitudes = Counter(magnitudes)

    lists = sorted(Group_magnitudes.items()) # sorted by key, return a list of tuples   

    x, y = zip(*lists) # unpack a list of pairs into two tuples
    somme=[y[0]]
    for count_mag in reversed(y[1:]) :
        somme = somme + [somme[-1]+count_mag]
    somme.reverse()
    print("somme -> ",somme)
       
    plt.plot(x, somme)
    plt.yscale('log')
    plt.savefig('Magnitude_Analyse.png')


df=pd.read_csv('../../data/cracks_X1Y2Z01_2k_granite_30MPa_r015.txt', sep= ' ')    

df.info()
Analyse_Magnitude(df)