"""
File where are written the classes used for the clustering process
"""


# Cluster
class Cluster:
    """
    Create a cluster from a list of rows taken fom a dataframe
    :param i_block: block where the cluster is created
    :param list_of_rows: rows inside the cluster
    :param block_size: size of a temporal block
    :param t_window: time window for the clustering process

    Attributes :
        - rows : rows inside the Cluster (i,p0,p1,p2,e)
        - transition_start : rows that are from the preceding block
        - transition_end : rows that are duplicated in the following block
    """

    def __init__(self, i_block, list_of_rows, block_size, t_window):
        """
        Create a cluster from a list of rows taken fom a dataframe
        :param i_block: block where the cluster is created
        :param list_of_rows: rows inside the cluster
        :param block_size: size of a temporal block
        :param t_window: time window for the clustering process
        """
        self.i_block = i_block
        self.rows = list_of_rows

        # List of elements that are from the preceding block
        self.transition_start = []
        for row in self.rows:
            if row.i < self.i_block * block_size:
                self.transition_start.append(row)
            else:
                break

        # List of the elements that are duplicated in the following block
        self.transition_end = []
        for row in reversed(self.rows):
            if row.i > (self.i_block + 1) * block_size - t_window:
                self.transition_end.append(row)
            else:
                break

    def can_be_merged(self):
        return len(self.transition_start) > 0 or len(self.transition_end) > 0

    def __repr__(self):
        representation = "Cluster:\n\ti_block: {},\n\tlen: {},".format(self.i_block, len(self.rows))
        for row in self:
            representation = representation + "\n\t{}".format(row)
        return representation

    def __iter__(self):
        return self.rows.__iter__()

    def __getitem__(self, item):
        return self.rows[item]

    def __len__(self):
        return len(self.rows)

    def __str__(self):
        i = self[0].i
        p0 = 0
        p1 = 0
        p2 = 0
        e = 0
        for crack in self:
            p0 = p0 + crack.p0
            p1 = p1 + crack.p1
            p2 = p2 + crack.p2
            e = e + crack.e
        p0 = p0/len(self)
        p1 = p1/len(self)
        p2 = p2/len(self)
        return "{},{},{},{},{}".format(i, p0, p1, p2, e)
