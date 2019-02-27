import pandas as pd
from projet.local_algorithms import cluster, analyse_magnitude, analyse_dimension, analyse_temporelle

if __name__ == '__main__':
    # Chargement des données
    df_in = pd.read_csv("../data/crack.csv", sep=" ")
    # print(df_in.head())

    # Cluster
    t_window = 200
    v_window = 0.25
    df_clusters = cluster(df_in, t_window, v_window)

    # Analyse
    print("# Analyse en magnitude #")
    print(analyse_magnitude(df_clusters))
    print("# Analyse en dimension #")
    print(analyse_dimension(df_clusters))
    print("# Analyse temporelle #")
    analyse_temporelle(df_clusters)


