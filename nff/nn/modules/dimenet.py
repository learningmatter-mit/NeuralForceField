import torch
from torch import nn

from nff.utils.scatter import scatter_add, compute_grad
from nff.utils.tools import layer_types
from nff.nn.layers import Dense


def get_dense(inp_dim, out_dim, activation, bias):
    """
    Create a dense layer.
    Args:
        inp_dim (int): dimension of input
        out_dim (int): dimension of output
        activation (str): name of activation layer
        bias (bool): whether or not to add a bias
    Returns:
        (nn.layers.Dense): dense layer
    """
    if activation is not None:
        activation = layer_types[activation]()
    return Dense(inp_dim, out_dim, activation=activation, bias=bias)


class EdgeEmbedding(nn.Module):

    """
    Class to create an edge embedding from edge features and
    node emebeddings.
    """

    def __init__(self, embed_dim, n_rbf, activation):
        """
        Args:
            embed_dim (int): embedding size
            n_rbf (int): number of radial basis functions
            activation (str): name of activation layer
        Returns:
            None
        """
        super().__init__()

        # create a dense layer that has input dimension
        # 3 * embed_dim (one each for h_i, h_j, and e_ij)
        # and output dimension embed_dim.

        self.dense = get_dense(
            3 * embed_dim,
            embed_dim,
            activation=activation,
            bias=True)

    def forward(self, h, e, nbr_list):
        """
        Call the model.
        Args:
            h (torch.Tensor): atomic embeddings
            e (torch.Tensor): edge feature vector
            nbr_list (torch.LongTensor): neighbor list
        Returns:
            m_ji (torch.Tensor): edge embedding vector.
        """
        m_ji = torch.cat((h[nbr_list[:, 0]], h[nbr_list[:, 1]], e), dim=-1)
        m_ji = self.dense(m_ji)
        return m_ji


class NodeEmbedding(nn.Module):
    """
    Class to generate node embeddings
    """

    def __init__(self, embed_dim):
        """
        Args:
            embed_dim (int): embedding size
        Returns:
            None
        """

        super().__init__()
        self.embedding = nn.Embedding(100, embed_dim, padding_idx=0)

    def forward(self, z):
        """
        Call the model.
        Args:
            z (torch.LongTensor): atomic numbers
        Returns:
            out (torch.Tensor): emebedding vector
        """
        out = self.embedding(z)
        return out


class EmbeddingBlock(nn.Module):
    """
    Block to create node and edge embeddings from coordinates
    and atomic numbers.

    """

    def __init__(self, n_rbf, embed_dim, activation):
        """
        Args:
            embed_dim (int): embedding size
            n_rbf (int): number of radial basis functions
            activation (str): name of activation layer
        Returns:
            None
        """
        super().__init__()
        # create a dense layer to convert the basis
        # representation of the distances into a vector
        # of size embed_dim
        self.edge_dense = get_dense(n_rbf,
                                    embed_dim,
                                    activation=None,
                                    bias=False)
        # make node and edge embedding layers
        self.node_embedding = NodeEmbedding(embed_dim)
        self.edge_embedding = EdgeEmbedding(embed_dim,
                                            n_rbf,
                                            activation)

    def forward(self, e_rbf, z, nbr_list):
        """
        Call the model.
        Args:
            e_rbf (torch.Tensor): radial basis representation
                of the distances
            z (torch.LongTensor): atomic numbers
            nbr_list (torch.LongTensor): neighbor list
        Returns:
            m_ji (torch.Tensor): edge embedding vector.
        """

        e = self.edge_dense(e_rbf)
        h = self.node_embedding(z)
        m_ji = self.edge_embedding(h=h,
                                   e=e,
                                   nbr_list=nbr_list)
        return m_ji


class ResidualBlock(nn.Module):
    """ Residual block """

    def __init__(self, embed_dim, n_rbf, activation):
        """
        Args:
            embed_dim (int): embedding size
            n_rbf (int): number of radial basis functions
            activation (str): name of activation layer
        Returns:
            None
        """

        super().__init__()
        # create dense layers
        self.dense_layers = nn.ModuleList(
            [get_dense(embed_dim,
                       embed_dim,
                       activation=activation,
                       bias=True)
                for _ in range(2)]
        )

    def forward(self, m_ji):
        """
        Args:
            m_ji (torch.Tensor): edge vector
        Returns:
            residual + m_ji (torch.Tensor):
                the edge vector plus the residual

        """

        residual = m_ji.clone()
        for layer in self.dense_layers:
            residual = layer(residual)

        return residual + m_ji


class DirectedMessage(nn.Module):

    """
    Module for passing directed messages based
    on distances and angles.
    """

    def __init__(self,
                 activation,
                 embed_dim,
                 n_rbf,
                 n_spher,
                 l_spher,
                 n_bilinear):
        """
        Args:
            activation (str): name of activation layer
            embed_dim (int): embedding size
            n_rbf (int): number of radial basis functions
            n_spher (int): maximum n value in the spherical
                basis functions
            l_spher (int): maximum l value in the spherical
                basis functions
            n_bilinear (int): dimension into which we will
                transform the n_spher * l_spher representation
                of angles and distances.
        Returns:
            None
        """

        super().__init__()

        # dense layer to apply to m's that are in the
        # neighborhood of those in your neighborhood
        self.m_kj_dense = get_dense(embed_dim,
                                    embed_dim,
                                    activation=activation,
                                    bias=True)

        # dense layer to apply to the rbf representation of
        # the distances
        self.e_dense = get_dense(n_rbf,
                                 embed_dim,
                                 activation=None,
                                 bias=False)

        # dense layer to apply to the sbf representation of
        # the angles and distances
        self.a_dense = get_dense(n_spher * l_spher,
                                 n_bilinear,
                                 activation=None,
                                 bias=False)

        # matrix that is used to aggregate the distance
        # and angle information
        self.w = nn.Parameter(torch.empty(
            embed_dim, n_bilinear, embed_dim))

        nn.init.xavier_uniform_(self.w)

    def forward(self,
                m_ji,
                e_rbf,
                a_sbf,
                kj_idx,
                ji_idx):
        """
        Args:
            m_ji (torch.Tensor): edge vector
            e_rbf (torch.Tensor): radial basis representation
                of the distances
            a_sbf (torch.Tensor): spherical basis representation
                of the distances and angles
            kj_idx (torch.LongTensor): nbr_list indices corresponding
                to the k,j indices in the angle list.
            ji_idx (torch.LongTensor): nbr_list indices corresponding
                to the j,i indices in the angle list.
        Returns:
            out (torch.Tensor): aggregated angle and distance information
                to be added to m_ji.
        """

        # apply the dense layers to e and m (ordered according to the kj
        # indices) and to a

        e_ji = self.e_dense(e_rbf[ji_idx])
        m_kj = self.m_kj_dense(m_ji[kj_idx])
        a = self.a_dense(a_sbf)

        # Defining e_m_kj = e_ji * m_kj and angle_len = len(angle_list),
        # this is equivalent to  torch.stack([torch.matmul(torch.matmul(
        # w, e_m_kj[i]), a[i]) for i in range(angle_len)]). So what we're
        # doing is multiplying a matrix w of dimension (embed x bilin x embed)
        # first by e_m_kj [vector of dimension (embed)], giving a matrix of
        # dimension (embed x bilin). Then we multiply by `a` [vector of
        # dimension (bilin)], giving a vector of dimension (embed). We repeat
        # this for all the kj neighbors. This gives us `aggr`, a matrix of
        # dimension (angle_len x embed), i.e. a vector of dimension (embed)
        # for each angle.

        aggr = torch.einsum("wj,wl,ijl->wi", a, m_kj * e_ji, self.w)

        # Now we want to sum each fingerprint aggr_ijk
        # over k. Say aggr = [aggr[angle_list[0]], aggr[angle_list[1]]]
        # =  [aggr_{0,1,2}, aggr_{0,1,3}]
        # = [aggr_{21, 10}, aggr_{31,10}].
        # The way we know the ji corresponding
        # to each aggr_kj,ji is by noting that they have
        # the same ordering as `angle_list`, and that the ji index of
        # each element in `angle_list` is given by `ji_idx`. Hence we
        # use `scatter_add` with indices `ji_idx`, and give the resulting
        # vector the same dimension as m_ji.

        out = scatter_add(aggr.transpose(0, 1),
                          ji_idx,
                          dim_size=m_ji.shape[0]
                          ).transpose(0, 1)

        return out


class DirectedMessagePP(nn.Module):
    def __init__(self,
                 activation,
                 embed_dim,
                 n_rbf,
                 n_spher,
                 l_spher,
                 int_dim,
                 basis_emb_dim):

        super().__init__()

        self.m_kj_dense = get_dense(embed_dim,
                                    embed_dim,
                                    activation=activation,
                                    bias=True)
        self.e_dense = nn.Sequential(get_dense(n_rbf,
                                               basis_emb_dim,
                                               activation=None,
                                               bias=False),
                                     get_dense(basis_emb_dim,
                                               embed_dim,
                                               activation=None,
                                               bias=False))

        self.a_dense = nn.Sequential(get_dense(n_spher * l_spher,
                                               basis_emb_dim,
                                               activation=None,
                                               bias=False),
                                     get_dense(basis_emb_dim,
                                               int_dim,
                                               activation=None,
                                               bias=False))

        self.down_conv = get_dense(embed_dim,
                                   int_dim,
                                   activation=activation,
                                   bias=False)

        self.up_conv = get_dense(int_dim,
                                 embed_dim,
                                 activation=activation,
                                 bias=False)

    def forward(self,
                m_ji,
                e_rbf,
                a_sbf,
                kj_idx,
                ji_idx):
        """
        Args:
            m_ji (torch.Tensor): edge vector
            e_rbf (torch.Tensor): radial basis representation
                of the distances
            a_sbf (torch.Tensor): spherical basis representation
                of the distances and angles
            kj_idx (torch.LongTensor): nbr_list indices corresponding
                to the k,j indices in the angle list.
            ji_idx (torch.LongTensor): nbr_list indices corresponding
                to the j,i indices in the angle list.
        Returns:
            out (torch.Tensor): aggregated angle and distance information
                to be added to m_ji.
        """

        e_ji = self.e_dense(e_rbf[ji_idx])
        m_kj = self.m_kj_dense(m_ji[kj_idx])
        a = self.a_dense(a_sbf)

        edge_message = self.down_conv(m_kj * e_ji)
        aggr = edge_message * a
        out = self.up_conv(scatter_add(aggr.transpose(0, 1),
                                       ji_idx,
                                       dim_size=m_ji.shape[0]
                                       ).transpose(0, 1))

        return out


class InteractionBlock(nn.Module):
    """
    Block for aggregating distance and angle information
    """

    def __init__(self,
                 embed_dim,
                 n_rbf,
                 activation,
                 n_spher,
                 l_spher,
                 n_bilinear,
                 int_dim=None,
                 basis_emb_dim=None,
                 use_pp=False):
        """
        Args:
            embed_dim (int): embedding size
            n_rbf (int): number of radial basis functions
            activation (str): name of activation layer
            n_spher (int): maximum n value in the spherical
                basis functions
            l_spher (int): maximum l value in the spherical
                basis functions
            n_bilinear (int): dimension into which we will
                transform the n_spher * l_spher representation
                of angles and distances.
        Returns:
            None
        """

        super().__init__()

        # make the three residual blocks
        self.residual_blocks = nn.ModuleList(
            [ResidualBlock(
                embed_dim=embed_dim,
                n_rbf=n_rbf,
                activation=activation) for _ in range(3)]
        )

        # make a block for getting the directed messages

        if use_pp:
            self.directed_block = DirectedMessagePP(
                activation=activation,
                embed_dim=embed_dim,
                n_rbf=n_rbf,
                n_spher=n_spher,
                l_spher=l_spher,
                int_dim=int_dim,
                basis_emb_dim=basis_emb_dim)

        else:
            self.directed_block = DirectedMessage(
                activation=activation,
                embed_dim=embed_dim,
                n_rbf=n_rbf,
                n_spher=n_spher,
                l_spher=l_spher,
                n_bilinear=n_bilinear)

        # dense layers for m_ji and for what comes after
        # the residual blocks
        self.m_ji_dense = get_dense(embed_dim,
                                    embed_dim,
                                    activation=activation,
                                    bias=True)

        self.post_res_dense = get_dense(embed_dim,
                                        embed_dim,
                                        activation=activation,
                                        bias=True)

    def forward(self,
                m_ji,
                e_rbf,
                a_sbf,
                kj_idx,
                ji_idx):
        """
        Args:
            m_ji (torch.Tensor): edge vector
            e_rbf (torch.Tensor): radial basis representation
                of the distances
            a_sbf (torch.Tensor): spherical basis representation
                of the distances and angles
            kj_idx (torch.LongTensor): nbr_list indices corresponding
                to the k,j indices in the angle list.
            ji_idx (torch.LongTensor): nbr_list indices corresponding
                to the j,i indices in the angle list.
        Returns:
            output (torch.Tensor): aggregated angle and distance information,
                combined with m_ji through residual blocks and skip
                connections.
        """

        # get the directed message
        directed_out = self.directed_block(m_ji=m_ji,
                                           e_rbf=e_rbf,
                                           a_sbf=a_sbf,
                                           kj_idx=kj_idx,
                                           ji_idx=ji_idx)
        # put m_ji through dense layer and add to directed
        # message
        dense_m_ji = self.m_ji_dense(m_ji)
        output = directed_out + dense_m_ji
        # put through one dense layer and add back m_ji
        output = self.post_res_dense(
            self.residual_blocks[0](output)) + m_ji
        # put through remaining dense layers
        for res_block in self.residual_blocks[1:]:
            output = res_block(output)

        return output


class OutputBlock(nn.Module):
    """
    Block to convert edge messages to atomic fingerprints
    """

    def __init__(self,
                 embed_dim,
                 n_rbf,
                 activation,
                 use_pp=False,
                 out_dim=None):
        """
        Args:
            embed_dim (int): embedding size
            n_rbf (int): number of radial basis functions
            activation (str): name of activation layer
        Returns:
            None
        """
        super().__init__()

        # dense layer to convert rbf edge representation
        # to dimension embed_dim
        self.edge_dense = get_dense(n_rbf,
                                    embed_dim,
                                    activation=None,
                                    bias=False)
        out_dense = []
        if use_pp:
            out_dense.append(get_dense(embed_dim,
                                       out_dim,
                                       activation=None,
                                       bias=False))
        else:
            out_dim = embed_dim

        out_dense += [get_dense(out_dim,
                                out_dim,
                                activation=activation,
                                bias=True)
                      for _ in range(3)]
        out_dense.append(get_dense(out_dim,
                                   out_dim,
                                   activation=None,
                                   bias=False))
        self.out_dense = nn.Sequential(*out_dense)

    def forward(self, m_ji, e_rbf, nbr_list, num_atoms):

        # product of e and m

        prod = self.edge_dense(e_rbf) * m_ji

        # Convert messages to node features.
        # The messages are m = {m_ji} =, for example,
        # [m_{0,1}, m_{0,2}, m_{1,0}, m{2,0}],
        # with nbr_list = [[0, 1], [0, 2], [1,0], [2,0]].
        # To sum over the j index we would have the first of
        # these messages add to index 1, the second to index 2,
        # and the last two to index 0. This means we use
        # nbr_list[:, 1] in the scatter addition.

        node_feats = scatter_add(prod.transpose(0, 1),
                                 nbr_list[:, 1],
                                 dim_size=num_atoms).transpose(0, 1)
        # Apply the dense layers
        node_feats = self.out_dense(node_feats)

        return node_feats
