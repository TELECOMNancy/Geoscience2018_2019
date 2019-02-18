from pyspark import SparkContext, SparkConf
from collections import namedtuple
from projet.spark_algorithms.cluster_class import Cluster
import numpy as np
import pandas as pd

# Constants
HEADER = ["i", "p0", "p1", "p2", "e"]
BLOCK_SIZE = 10000
V_WINDOW = 0.25
T_WINDOW = 2000

# Crack
Crack = namedtuple('Crack', HEADER)


def parse(line):
    line_splited = line.split(" ")
    return Crack(np.int(line_splited[0]),
                 np.float(line_splited[1]),
                 np.float(line_splited[2]),
                 np.float(line_splited[3]),
                 np.float(line_splited[9]),
                 )


def map_before_block(crack: Crack):
    i_bloc = int(crack.i / BLOCK_SIZE)
    to_duplicate = BLOCK_SIZE * (i_bloc + 1) - crack.i <= T_WINDOW

    if to_duplicate:
        return [
            (i_bloc, [crack]),
            (i_bloc + 1, [crack])
        ]
    else:
        return [
            (i_bloc, [crack]),
        ]


def reduce_in_block(list1, list2):
    list1.extend(list2)
    return list1


def map_cluster(tuple):
    i_block = tuple[0]
    list_of_cracks = tuple[1]
    list_of_cracks = sorted(list_of_cracks, key=lambda crack: crack.i)

    df_cluster = pd.DataFrame(list_of_cracks, columns=HEADER)

    delta_t = T_WINDOW
    delta_v = V_WINDOW ** 2

    number_of_rows = len(df_cluster.index)
    clustering_list = [-1] * number_of_rows

    last_first_correct_element = 0
    for row_i in df_cluster.itertuples():
        index_i = row_i.Index
        # If the element is already in a cluster we get his cluster_number else we create a new cluster_number
        current_cluster = clustering_list[index_i] if clustering_list[index_i] != -1 else max(clustering_list) + 1
        in_the_temporal_window: bool = False
        # The elements are sorted on their iteration number, once we have surpassed the time frame
        # the following elements won't be of use
        # If was_in_time is False, we are before the temporal window
        for row_j in df_cluster.itertuples():
            index_j = row_j.Index
            if index_j >= last_first_correct_element:
                # delta_t_ij = compute_time_distance(row_i, row_j)
                delta_t_ij = abs(row_i.i - row_j.i)
                # if the point is in the temporal window
                if delta_t_ij <= delta_t:
                    if not in_the_temporal_window:
                        last_first_correct_element = index_j
                        in_the_temporal_window = True
                    # delta_v_ij = compute_euclidean_distance_square(row_i, row_j)
                    delta_v_ij = (row_i.p0 - row_j.p0) ** 2 + (row_i.p1 - row_j.p1) ** 2 + (row_i.p2 - row_j.p2) ** 2
                    # If there is a collocation in time and space
                    if delta_v_ij <= delta_v:
                        # If the crack has already a cluster assigned and it is different from the current cluster
                        if clustering_list[index_j] != -1 and clustering_list[index_j] != current_cluster:
                            # Merge the two clusters
                            cluster_to_merge = clustering_list[index_j]
                            for index in range(len(clustering_list)):
                                if clustering_list[index] == cluster_to_merge:
                                    clustering_list[index] = current_cluster
                        else:
                            clustering_list[index_j] = current_cluster
                # elif the point is after the temporal window
                elif in_the_temporal_window:
                    break

    df_cluster["cluster"] = clustering_list

    clusters: list = []

    for name, group in df_cluster.groupby("cluster"):
        # i of the cluster is equal to the first i
        group = group.drop(columns=["cluster"])
        clusters.append(Cluster(i_block, list(group.itertuples(index=False, name="Crack")), block_size=BLOCK_SIZE, t_window=T_WINDOW))

    return clusters


if __name__ == '__main__':
    print("Spark is starting")
    # Test file in input
    data_test = "../../data/test2.txt"  # Should be some file on your system
    # Variables
    BLOCK_SIZE = 10
    V_WINDOW = 1
    T_WINDOW = 1
    # Configuration of the spark work
    conf = (SparkConf()
            .setMaster("local")
            .setAppName("Clustering")
            )
    sc = SparkContext(conf=conf)
    # Spark work
    print("Spark is working")
    rdd_input = sc.textFile(data_test).filter(lambda x: not x.startswith("i"))
    rdd_parsed = rdd_input.map(parse)
    # t1 = rdd_parsed.first()
    rdd_in_block = rdd_parsed.flatMap(map_before_block).reduceByKey(reduce_in_block)
    rdd_clustered = rdd_in_block.flatMap(map_cluster)
    # t2 = rdd_clustered.first()[0]
    print(rdd_clustered.first())
    print("SPark has finished its work")
