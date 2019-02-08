import pandas as pd


def compute_time_distance(row_i, row_j):
    return abs(row_i['i'] - row_j['i'])


def compute_euclidean_distance_square(row_i, row_j):
    """
    Calculate the squared euclidean distance between two point represent by rows
    We don't have to calculate the square root, which is an expensive operation
    :param row_i: first row
    :param row_j: second row
    :return:
    """
    # point1 = (row_i["p0"], row_i["p1"], row_i["p2"])
    # point2 = (row_j["p0"], row_j["p1"], row_j["p2"])
    return (row_i["p0"] - row_j["p0"]) ** 2 + (row_i["p1"] - row_j["p1"]) ** 2 + (row_i["p2"] - row_j["p2"]) ** 2


def cluster(df_in: pd.DataFrame, delta_t: float, delta_v: float):
    """
    Create a new dataframe which is constructed via the clustering of df_in
    :param df_in:
    :param delta_t: temporal frame
    :param delta_v: spacial frame
    :return: A new dataframe
    """

    # As we don't compute the square root
    delta_v = delta_v ** 2

    number_of_rows = len(df_in.index)
    clustering_list = [-1] * number_of_rows

    print("Creation of clusters")
    last_first_correct_element = 0
    # We save the start of the temporal frame of the precedent iteration
    for index_i, row_i in df_in.iterrows():
        # If the element is already in a cluster we get his cluster_number else we create a new cluster_number
        print(index_i, "/", number_of_rows)
        current_cluster = clustering_list[index_i] if clustering_list[index_i] != -1 else max(clustering_list) + 1
        was_in_time: bool = False
        # The elements are sorted on their iteration number, once we have surpassed the time frame
        # the following elements won't be of use
        for index_j, row_j in df_in.iterrows():
            if index_j >= last_first_correct_element:
                delta_t_ij = compute_time_distance(row_i, row_j)
                if delta_t_ij < delta_t:
                    if not was_in_time:
                        last_first_correct_element = index_j
                    was_in_time = True
                    delta_v_ij = compute_euclidean_distance_square(row_i, row_j)
                    # If there is a collocation in time and space
                    if delta_v_ij < delta_v:
                        if clustering_list[index_j] != -1 and clustering_list[index_j] != current_cluster:
                            # Merge the two clusters
                            cluster_to_merge = clustering_list[index_j]
                            for index in range(clustering_list):
                                if clustering_list[index] == cluster_to_merge:
                                    clustering_list[index] = current_cluster
                        else:
                            clustering_list[index_j] = current_cluster
                elif was_in_time:
                    break

    df_clustering = df_in
    df_clustering["cluster"] = clustering_list

    print("Formalize the result")
    # create the new data
    data_out = {
        "i": [],
        "p0": [],
        "p1": [],
        "p2": [],
        "t": [],
        "e": [],
    }

    for name, group in df_clustering.groupby("cluster"):
        # i of the cluster is equal to the first i
        i = group.iloc[0, :]["i"]
        p0 = 0
        p1 = 0
        p2 = 0
        # t of the cluster is equal to the first t
        t = group.iloc[0, :]["t"]
        e = 0
        for index, row in group.iterrows():
            p0 += row["p0"]
            p1 += row["p1"]
            p2 += row["p2"]
            e += row["e"]

        count = len(group.index)
        p0 = p0 / count
        p1 = p1 / count
        p2 = p2 / count

        data_out["i"].append(i)
        data_out["p0"].append(p0)
        data_out["p1"].append(p1)
        data_out["p2"].append(p2)
        data_out["t"].append(t)
        data_out["e"].append(e)

    return pd.DataFrame.from_dict(data_out)


if __name__ == "__main__":
    df_file: pd.DataFrame = pd.read_csv("../../data/test.txt", sep=" ")
    #df_file: pd.DataFrame = pd.read_csv("../../data/cracks_X1Y2Z01_2k_granite_30MPa_r015.txt", sep=" ")
    print("__main__ -> \n", df_file.head(5))
    df_res = cluster(df_file, 200, 5)
    print("__res__ ->\n", df_res.head(5))
