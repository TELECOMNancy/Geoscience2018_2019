import pandas as pd
from console_progressbar import ProgressBar


# def compute_time_distance(row_i, row_j):
#     """
#     :param row_i: named_tuple of the first row
#     :param row_j: named_tuple of the second row
#     :return:
#     """
#     return abs(row_i.i - row_j.i)


# def compute_euclidean_distance_square(row_i, row_j):
#     """
#     Calculate the squared euclidean distance between two point represent by rows
#     We don't have to calculate the square root, which is an expensive operation
#     :param row_i: named tuple of the first row
#     :param row_j: second row of the second row
#     :return:
#     """
#     # point1 = (row_i["p0"], row_i["p1"], row_i["p2"])
#     # point2 = (row_j["p0"], row_j["p1"], row_j["p2"])
#     # return (row_i["p0"] - row_j["p0"]) ** 2 + (row_i["p1"] - row_j["p1"]) ** 2 + (row_i["p2"] - row_j["p2"]) ** 2
#     return (row_i.p0 - row_j.p0) ** 2 + (row_i.p1 - row_j.p1) ** 2 + (row_i.p2 - row_j.p2) ** 2


def cluster(df_in: pd.DataFrame, delta_t: float, delta_v: float):
    """
    Create a new dataframe which is constructed via the clustering of df_in
    :param df_in:
    :param delta_t: temporal frame
    :param delta_v: spacial frame
    :return: A new dataframe
    """

    df_cluster = df_in[["i", "p0", "p1", "p2", "e"]]

    # As we don't compute the square root
    delta_v = delta_v ** 2

    number_of_rows = len(df_cluster.index)
    clustering_list = [-1] * number_of_rows

    # We save the start of the temporal frame of the precedent iteration
    pb = ProgressBar(total=number_of_rows, prefix='Creation of clusters', suffix='', decimals=0, length=50, fill='-', zfill=' ')
    last_first_correct_element = 0
    for index_i, row_i in enumerate(df_cluster.itertuples()):
        pb.print_progress_bar(index_i)
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
                    delta_v_ij = (row_i.p0 - row_j.p0) ** 2  + (row_i.p1 - row_j.p1) ** 2 + (row_i.p2 - row_j.p2) ** 2
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

    print(df_cluster.head(15))

    print("Formalize the result")
    # create the new data
    data_out = {
        "i": [],
        "p0": [],
        "p1": [],
        "p2": [],
        "e": [],
    }

    for name, group in df_cluster.groupby("cluster"):
        # i of the cluster is equal to the first i
        i = group.iloc[0, :]["i"]
        p0 = 0
        p1 = 0
        p2 = 0
        # t of the cluster is equal to the first t
        e = 0
        for row in group.itertuples():
            p0 = p0 + row.p0
            p1 = p1 + row.p1
            p2 = p2 + row.p2
            e = e + row.e

        count = len(group.index)
        p0 = p0 / count
        p1 = p1 / count
        p2 = p2 / count

        data_out["i"].append(i)
        data_out["p0"].append(p0)
        data_out["p1"].append(p1)
        data_out["p2"].append(p2)
        data_out["e"].append(e)

    print(data_out["i"][:15])

    return pd.DataFrame.from_dict(data_out)


def test():
    # df_file: pd.DataFrame = pd.read_csv("../../data/test.txt", sep=" ")
    df_file: pd.DataFrame = pd.read_csv("../../data/cracks_X1Y2Z01_2k_granite_30MPa_r015.txt", sep=" ")
    t_window = 200
    v_window = 0.25
    print("__main__ -> \n", df_file.head(15))
    df_res = cluster(df_file, t_window, v_window)
    print("__res__ ->\n", df_res.head(5))
    df_res.to_csv("../../data/clustering_results_t_" + str(t_window) + "_v_" + str(v_window).replace(".", "point")
                  + ".csv", index=False)


if __name__ == "__main__":
    import time
    a = time.time()
    test()
    b = time.time()
    print("TIME : ", b-a, "s")
