from pyspark import SparkContext, SparkConf
from collections import namedtuple
from projet.spark_algorithms.cluster_class import Cluster
import numpy as np
import pandas as pd
import sys
import time

# Arguments
ARGUMENTS = {
    "input_file": 1,
    "output_file": 2,
    "block_size": 3,
    "t_window": 4,
    "v_window": 5,
}

# Constants
HEADER = ["i", "p0", "p1", "p2", "e"]

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


def map_before_reduce_in_block(crack: Crack):
    i_bloc = int(crack.i / BLOCK_SIZE)
    return i_bloc, [crack]


def reduce_in_block(list1, list2):
    list1.extend(list2)
    return list1


def map_cluster(tuple_):
    i_block = tuple_[0]
    list_of_cracks = tuple_[1]
    list_of_cracks = sorted(list_of_cracks, key=lambda crack: crack.i)

    df_cluster = pd.DataFrame(list_of_cracks, columns=HEADER)

    delta_t = T_WINDOW
    delta_v = V_WINDOW ** 2

    number_of_rows = len(df_cluster.index)
    clustering_list = [-1] * number_of_rows

    last_first_correct_element = 0
    for index_i, row_i in enumerate(df_cluster.itertuples()):
        # If the element is already in a cluster we get his cluster_number else we create a new cluster_number
        current_cluster = clustering_list[index_i] if clustering_list[index_i] != -1 else max(clustering_list) + 1
        in_the_temporal_window: bool = False
        # The elements are sorted on their iteration number, once we have surpassed the time frame
        # the following elements won't be of use
        # If was_in_time is False, we are before the temporal window
        for index_j, row_j in enumerate(df_cluster.itertuples()):
            if index_j >= last_first_correct_element:
                # delta_t_ij = compute_time_distance(row_i, row_j)
                delta_t_ij = abs(row_i.i - row_j.i)
                # if the point is in the temporal window
                if delta_t_ij < delta_t:
                    if not in_the_temporal_window:
                        last_first_correct_element = index_j
                        in_the_temporal_window = True
                    # delta_v_ij = compute_euclidean_distance_square(row_i, row_j)
                    delta_v_ij = (row_i.p0 - row_j.p0) ** 2 + (row_i.p1 - row_j.p1) ** 2 + (row_i.p2 - row_j.p2) ** 2
                    # If there is a collocation in time and space
                    if delta_v_ij < delta_v:
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

    # if i_block == 1:
    #     print(df_cluster.head(15))

    clusters: list = []

    for name, group in df_cluster.groupby("cluster"):
        # i of the cluster is equal to the first i
        group = group.drop(columns=["cluster"])
        clusters.append(
            Cluster(
                i_block,
                list(group.itertuples(index=False, name="Crack")),
                block_size=BLOCK_SIZE,
                t_window=T_WINDOW
            )
        )

    return clusters


if __name__ == '__main__':
    print(sys.argv)
    if len(sys.argv) != len(ARGUMENTS) + 1:
        print("USAGE python clustering_parallel.py <input_file> <output_file> <block_size> <t_window> <v_window>")

    else:
        print("Parsing the input")
        time_start = time.time()
        # Parsing input
        data_input = sys.argv[ARGUMENTS["input_file"]]  # Should be some file on your system
        data_output = sys.argv[ARGUMENTS["output_file"]]
        BLOCK_SIZE = int(sys.argv[ARGUMENTS["block_size"]])
        V_WINDOW = float(sys.argv[ARGUMENTS["v_window"]])
        T_WINDOW = int(sys.argv[ARGUMENTS["t_window"]])
        print("Starting spark")
        # Configuration of the spark work
        conf = (SparkConf()
                .setMaster("local")
                .setAppName("Clustering")
                )
        sc = SparkContext(conf=conf)
        # Spark work
        print("Spark is working")

        rdd_input = sc.textFile(data_input).filter(lambda x: not x.startswith("i"))

        rdd_parsed = rdd_input.map(parse)
        rdd_in_block = rdd_parsed.map(map_before_reduce_in_block).reduceByKey(reduce_in_block)
        rdd_clustered = rdd_in_block.flatMap(map_cluster).cache()

        # Merge of the clusters
        print("Merging the clusters")
        rdd_clustered_need_to_merge = rdd_clustered.filter(lambda cluster: cluster.can_be_merged())
        list_clusters_merged = list(rdd_clustered_need_to_merge.collect())

        index_1: int = 0
        while index_1 < len(list_clusters_merged):
            cluster_1 = list_clusters_merged[index_1]

            index_2: int = index_1 + 1
            has_been_merged: bool = False
            while index_2 < len(list_clusters_merged):
                cluster_2 = list_clusters_merged[index_2]
                if cluster_1.can_be_merged_with(cluster_2, T_WINDOW, V_WINDOW):
                    has_been_merged = True
                    new_cluster = Cluster.merge(cluster_1, cluster_2)
                    # Remove the old cluster and add the new one
                    list_clusters_merged.remove(cluster_1)
                    list_clusters_merged.remove(cluster_2)
                    list_clusters_merged.insert(index_1, new_cluster)
                    break
                index_2 = index_2 + 1

            if not has_been_merged:
                index_1 = index_1 + 1

        # Save the clusters
        rdd_final = rdd_clustered.filter(lambda cluster: not cluster.can_be_merged())
        rdd_merged = sc.parallelize(list_clusters_merged)
        rdd_final = rdd_final.union(rdd_merged)

        rdd_final_mapped = rdd_final.map(lambda cluster: cluster.__str__())
        rdd_final_mapped.saveAsTextFile(data_output)

        time_end = time.time()

        print("Spark has finished its work in : ", time_end-time_start, "s")
        print("The result has been saved in ", data_output)
        print("Final number of clusters :", rdd_final_mapped.count())
