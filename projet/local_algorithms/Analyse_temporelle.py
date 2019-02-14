import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from math import gamma
from sklearn.preprocessing import MinMaxScaler

#df=pd.read_csv(
#'/Users/sakina/Downloads/cracks_X1Y2Z01_2k_granite_30MPa_r015.txt', sep= ' ')


# Analyse temporelle 
# Loi d'Omori 

# Loi omori simplifiée 
def simple_omori(pas_de_temps, t ):
    t_declenchement_microcrack = t * pas_de_temps 
    
    # Normalisation de t
    min = np.min( t_declenchement_microcrack)
    max = np.max(t_declenchement_microcrack)
    t_normalized = t_declenchement_microcrack * len(t_declenchement_microcrack) / (max - min)

    Nt = 1/t_declenchement_microcrack
    #print(t_declenchement_microcrack.head())
    
# Loi omori (Utsu) 
def omori(pas_de_temps, t , K, p_value, c):
    t_declenchement_microcrack = t * pas_de_temps
    Nt = K/np.power(t_declenchement_microcrack + c , p_value)
    print(Nt.head())

    # Normalisation de t
    min = np.min( t_declenchement_microcrack)
    max = np.max(t_declenchement_microcrack)
    t_normalized = t_declenchement_microcrack * len(t_declenchement_microcrack) / (max - min)

# Calcul densité de probabilité (loi Gamma)
def proba(t_normalized):
    b = np.var(t_normalized)/np.mean(t_normalized)
    print("beta " , b)
    gam = np.mean(t_normalized)/b
    print("gamma " , gam)
    return C(gam, b) * np.power(t_normalized, gam - 1) * np.exp(-t_normalized / b)

def C(gam, b):
    C =  np.power(b, gam) * gamma(gam)
    return 1/C

def analyse_temporelle(df):
    t_declenchement_microcrack = df['i'] * 0.0001 

    dif = np.diff(t_declenchement_microcrack)
    # Supprimer les valeurs nulles
    dif = dif[dif!=0]
    # Tri de dif
    dif = np.sort(dif)

    min = np.min(dif)
    max = np.max(dif)
    # Normalisation de t
    lamda = len(dif) / (max - min)
    t_normalized = dif * lamda

    p = proba(t_normalized)

    fig, ax = plt.subplots()
    plt.plot(  t_normalized, p)
    plt.plot(  t_normalized, 1/t_normalized)
    plt.plot(  t_normalized, np.exp(-t_normalized))
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('Temps normalisé')
    plt.ylabel('Densité de probabilité temporelle')
    plt.grid()
    fig.savefig("test6.png")
    plt.show()