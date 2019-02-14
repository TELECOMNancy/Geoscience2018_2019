import pandas as pd
from local_algorithms import cluster, analyse_magnitude

if __name__ == '__main__':
    # Chargement des données
    df_in = pd.read_csv("../data/cracks_X1Y2Z01_2k_granite_30MPa_r015.txt", sep=" ")
    # print(df_in.head())

    # Cluster
    t_window = 200
    v_window = 0.25
    df_clusters = cluster(df_in, t_window, v_window)

    # Analyse
    print("# Analyse en magnitude #")
    print(analyse_magnitude(df_clusters))


