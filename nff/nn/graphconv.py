import torch.nn as nn
from nff.utils.scatter import scatter_add


class MessagePassingModule(nn.Module):

    """Convolution constructed as MessagePassing.
    """

    def __init__(self):
        super(MessagePassingModule, self).__init__()

    def message(self, r, e, a, aggr_wgt):
        # Basic message case
        assert r.shape[-1] == e.shape[-1]
        # mixing node and edge feature, multiply by default
        # possible options:
        # (ri [] eij) -> rj,
        # where []: *, +, (,), permutation....
        if aggr_wgt is not None:
            r = r * aggr_wgt

        message = r[a[:, 0]] * e, r[a[:, 1]] * e
        return message

    def aggregate(self, message, index, size):
        # pdb.set_trace()
        new_r = scatter_add(src=message,
                            index=index,
                            dim=0,
                            dim_size=size)
        return new_r

    def update(self, r):
        return r

    def forward(self, r, e, a, aggr_wgt=None):

        graph_size = r.shape[0]

        rij, rji = self.message(r, e, a, aggr_wgt)
        # i -> j propagate
        r = self.aggregate(rij, a[:, 1], graph_size)
        # j -> i propagate
        r += self.aggregate(rji, a[:, 0], graph_size)
        r = self.update(r)
        return r


class EdgeUpdateModule(nn.Module):
    """Update Edge State Based on information from connected nodes
    """

    def __init__(self):
        super(EdgeUpdateModule, self).__init__()

    def message(self, r, e, a):
        """Summary

        Args:
            r (TYPE): node vectors
            e (TYPE): edge vectors
            a (TYPE): neighbor list

        Returns:
            TYPE: Description
        """
        message = r
        return message

    def aggregate(self, message, neighborlist):
        """aggregate function that aggregates information from
            connected nodes

        Args:
            message (TYPE): Description
            neighborlist (TYPE): Description

        Returns:
            TYPE: Description
        """
        aggregated_edge_feature = message[neighborlist[:, 0]
                                          ] + message[neighborlist[:, 1]]
        return aggregated_edge_feature

    def update(self, e):
        return e

    def forward(self, r, e, a):
        message = self.message(r, e, a)
        # update edge from two connected nodes
        e = self.aggregate(message, a)
        e = self.update(e)
        return e


class GeometricOperations(nn.Module):

    """Compute geomtrical properties based on XYZ coordinates
    """

    def __init__(self):
        super(GeometricOperations, self).__init__()


class TopologyOperations(nn.Module):

    """Change the topology index given geomtrical properties
    """

    def __init__(self):
        super(TopologyOperations, self).__init__()
