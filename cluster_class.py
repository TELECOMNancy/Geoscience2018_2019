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

    def __init__(self, i_block=None, list_of_rows=None, block_size=None, t_window=None):
        """
        Create a cluster from a list of rows taken fom a dataframe
        :param i_block: block where the cluster is created
        :param list_of_rows: rows inside the cluster
        :param block_size: size of a temporal block
        :param t_window: time window for the clustering process
        """

        if i_block is None and list_of_rows is None and block_size is None and t_window is None:
            self.i_block = []
            self.rows = []
            self.transition_start = []
            self.transition_end = []

        else:
            self.i_block = [i_block]
            self.rows = list_of_rows

            # List of elements that are from the preceding block
            self.transition_start = []
            for row in self.rows:
                if row.i <= self.i_block[0] * block_size + t_window:
                    self.transition_start.append(row)
                else:
                    break

            # List of the elements that are duplicated in the following block
            self.transition_end = []
            for row in reversed(self.rows):
                if row.i >= (self.i_block[0] + 1) * block_size - t_window:
                    self.transition_end.append(row)
                else:
                    break

    def can_be_merged(self):
        return len(self.transition_start) > 0 or len(self.transition_end) > 0

    # Return true if two cluster are to be merged
    def can_be_merged_with(self, other, t_window, v_window):
        v_window = v_window ** 2

        neighbour_self = []
        for i_block in self.i_block:
            neighbour_self.append(i_block - 1)
            neighbour_self.append(i_block)
            neighbour_self.append(i_block + 1)

        # If the cluster aren't neighbours, we don't have to check if they can be merged
        if len(set(neighbour_self).intersection(set(other.i_block))) == 0:
            return False

        # Then we search if points are in t_window and v_window

        # start-end
        for points_from_self in self.transition_start:
            for points_from_other in other.transition_end:
                delta_t = abs(points_from_self.i - points_from_other.i)
                if delta_t < t_window:
                    delta_v = (points_from_self.p0 - points_from_other.p0) ** 2 + (
                                points_from_self.p1 - points_from_other.p1) ** 2 + (
                                          points_from_self.p2 - points_from_other.p2) ** 2
                    if delta_v < v_window:
                        return True

        # end-start
        for points_from_self in self.transition_end:
            for points_from_other in other.transition_start:
                delta_t = abs(points_from_self.i - points_from_other.i)
                if delta_t < t_window:
                    delta_v = (points_from_self.p0 - points_from_other.p0) ** 2 + (
                                points_from_self.p1 - points_from_other.p1) ** 2 + (
                                          points_from_self.p2 - points_from_other.p2) ** 2
                    if delta_v < v_window:
                        return True

        # start - start
        for points_from_self in self.transition_start:
            for points_from_other in other.transition_start:
                delta_t = abs(points_from_self.i - points_from_other.i)
                if delta_t < t_window:
                    delta_v = (points_from_self.p0 - points_from_other.p0) ** 2 + (
                                points_from_self.p1 - points_from_other.p1) ** 2 + (
                                          points_from_self.p2 - points_from_other.p2) ** 2
                    if delta_v < v_window:
                        return True

        # end-start
        for points_from_self in self.transition_end:
            for points_from_other in other.transition_end:
                delta_t = abs(points_from_self.i - points_from_other.i)
                if delta_t < t_window:
                    delta_v = (points_from_self.p0 - points_from_other.p0) ** 2 + (
                                points_from_self.p1 - points_from_other.p1) ** 2 + (
                                          points_from_self.p2 - points_from_other.p2) ** 2
                    if delta_v < v_window:
                        return True

    @classmethod
    def merge(cls, cluster_1, cluster_2):
        new_cluster = Cluster()
        new_cluster.i_block = cluster_1.i_block + cluster_2.i_block
        new_cluster.rows = cluster_1.rows + cluster_2.rows
        new_cluster.transition_start = cluster_1.transition_start + cluster_2.transition_start
        new_cluster.transition_end = cluster_1.transition_end + cluster_2.transition_end
        return new_cluster


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
        p0 = p0 / len(self)
        p1 = p1 / len(self)
        p2 = p2 / len(self)
        return "{},{},{},{},{}".format(i, p0, p1, p2, e)
