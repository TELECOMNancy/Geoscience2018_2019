import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler

df=pd.read_csv(
'/Users/sakina/Downloads/cracks_X1Y2Z01_2k_granite_30MPa_r015.txt', sep= ' ')

df.info()

# Analyse temporelle 
# Loi d'Omori 

p_val= 1
pas_de_temps = 0.0001
i= df['i']
t_declenchement_microcrack= i*pas_de_temps 
Nt = 1/t_declenchement_microcrack

#print(t_declenchement_microcrack.head())
min = np.min( t_declenchement_microcrack)
max = np.max(t_declenchement_microcrack)

#print(min, max)
t_normalized = (t_declenchement_microcrack - min) / (max - min)

print(t_normalized.head())

#print(' t_declenchement_microcrack  ' , t_declenchement_microcrack) 
plt.plot(Nt, t_normalized)
plt.xlabel('Temps')
plt.ylabel('Densité de probabilité temporelle')
plt.show()