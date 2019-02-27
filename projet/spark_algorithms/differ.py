import pandas as pd

if __name__ == "__main__":

    df1 = pd.read_csv("../../clusters/no_need_to_merge_formatted/part-00000"
                , names=["i", "p0", "p1", "p2", "e"]).round(3)

    df2 = pd.read_csv("../../data/clustering_results_t_200_v_0point25.csv").round(3)

    s1 = pd.merge(df1, df2.round(3), how='inner', on=['i','p0', 'p1', 'p2', 'e'])

    s2 = df1[~df1.isin(df2)].dropna()
    print(s2.head())