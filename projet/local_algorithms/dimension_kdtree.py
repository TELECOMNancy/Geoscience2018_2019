from scipy import spatial
import numpy as np
import matplotlib.pyplot as plt

''''
x, y = np.mgrid[0:5, 0:5]
points = np.c_[x.ravel(), y.ravel()]
tree = spatial.KDTree(points)
tree.query_ball_point([2, 0], 1)

'''




########################### ANALYSE SPATIALE ##################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

# 'nrows = nb_ligne' en 3ème argument pour spécifier le nombre de ligne dans read_table






def get_all_position(p0,p1,p2):
    position = list()
    for i in range(0,len(p0)):
        position.append([p0[i],p1[i],p2[i]])
    return np.asarray(position)



def create_tree(positions):
    tree = spatial.KDTree(positions)
    return tree


def pairs_number(clusters_tab):
    ''' 
        Fonction qui calcule le nombre de paires de cluster(somme de 1 à n-1)
        arg 0 : tableau des clusters 
    '''
    return (len(clusters_tab)-1)*len(clusters_tab)/2

def eulerian_distance(a,b):
    ''' 
        Fonction qui calcule la distance entre deux points dans l'espace 
        arg 0 : point a
        arg 1 ; point b 
    '''
    return ((b[0]-a[0])**2 + (b[1]-a[1])**2 + (b[2]-a[2])**2)**0.5

def near_pairs_number_optimised(tree,distance):
    ''' 
        Fonction qui calcule le nombre de paires de cluster dont les barycentres
        sont espacés d'une distance inférieure à 'distance' 
        arg 0 : tableau des clusters
        arg 1 : distance
    '''
    return len(tree.query_pairs(distance))
           
def get_distance_min_max(clusters_tab):
    distance_min = 100000
    distance_max = 0
    i = 1
    for cluster in clusters_tab[:-1]:
        for j in range(i,len(clusters_tab)):
            if eulerian_distance(cluster,clusters_tab[j]) <= distance_min :
                distance_min = eulerian_distance(cluster,clusters_tab[j])
            if eulerian_distance(cluster,clusters_tab[j]) > distance_max :
                distance_max = eulerian_distance(cluster,clusters_tab[j])
        i +=1
    return distance_min, distance_max 
    

def get_abcisses(nb,minimum,maximum):
    #minimum_log = math.log(minimum)
    #maximum_log = math.log(maximum)
    return np.logspace(minimum, maximum,num=nb)

def get_ordonnees(tree,abcisse,nb_total):
    y = list()
    compteur = 0
    maximum = 0
    compteur_pas = 0
    for i in range(0,len(abcisse)):
        near_pairs = math.log(near_pairs_number_optimised(tree,abcisse[i]))/nb_total
        if maximum != near_pairs:
            y.append(near_pairs)
            maximum = near_pairs
            compteur +=1
            compteur_pas +=1
            print(str(compteur/len(abcisse)*100)+'%')
        else : break
    return y,compteur_pas
        
def recompute_abcisses(x,nb):
    x = x[:nb]
    return x
        
def analyse_dimension(df):
    p0 = df['p0']
    p1 = df['p1']
    p2 = df['p2']

    position = get_all_position(p0,p1,p2)
    tree = create_tree(position)
    nt = pairs_number(position)
    minimum, maximum = get_distance_min_max(position)


    x = get_abcisses(20,minimum,maximum)
    y, compteur = get_ordonnees(tree,x,nt)
    x = recompute_abcisses(x,compteur)
    plt.figure()
    plt.xlabel('distance log(r) ')
    plt.ylabel('corrélation spatiale')
    plt.title('Graphe de corrélation spatiale')
    plt.xscale('log')
    plt.plot(x, y)

    fit = np.polyfit(x, y, 1) 

if __name__ == '__main__':
    df = pd.read_table("data/cracks_X1Y2Z01_2k_granite_30MPa_r015.txt", sep=' ', header=0)
    analyse_dimension(df)




















































