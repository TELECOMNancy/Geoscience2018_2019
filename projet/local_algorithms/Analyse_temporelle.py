import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from math import gamma
from sklearn.preprocessing import MinMaxScaler

df=pd.read_csv(
'/Users/sakina/Downloads/cracks_X1Y2Z01_2k_granite_30MPa_r015.txt', sep= ' ')

#df.info()

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
    b = beta(t_normalized)
    return C(b) * np.power(t_normalized, 1/b - 1) * np.exp(-t_normalized / b)

def C(b):
    return 1/(np.power(b, 1/b) * gamma(1/b))
    
def beta(t_normalized):
    return np.var(t_normalized) / t_normalized


t_declenchement_microcrack = df['i'] * 0.0001 
# Normalisation de t
min = np.min( t_declenchement_microcrack)
max = np.max(t_declenchement_microcrack)
t_normalized = t_declenchement_microcrack * len(t_declenchement_microcrack) / (max - min)

#print(proba(t_normalized).head())

'''
fig, ax = plt.subplots()
plt.plot( proba(t_normalized) , t_normalized)
plt.xlabel('Temps normalisé')
plt.ylabel('Densité de probabilité temporelle')
plt.grid()
plt.xscale('log',basey=10) 
plt.yscale('log',basey=10) 
fig.savefig("test3.png")
plt.show() '''

print(C(beta(t_normalized))