import torch
from torch import nn

from nff.nn.modules.dimenet import (EmbeddingBlock, InteractionBlock,
                                    OutputBlock)
from nff.nn.layers import DimeNetRadialBasis as RadialBasis
from nff.nn.layers import DimeNetSphericalBasis as SphericalBasis
from nff.utils.scatter import compute_grad


def compute_angle(xyz, angle_list):
    """
    Compute angles between atoms.
    Args:
        xyz (torch.Tensor): coordinates of the atoms.
        angle_list (torch.LongTensor): directed indices
            of sets of three atoms that are all in each
            other's neighborhood.
    Returns:
        angle (torch.Tensor): tensor of angles for each
            element in the angle list.

    """

    # points from j -> i
    r_ji = xyz[angle_list[:, 0]] - xyz[angle_list[:, 1]]
    # points from j -> k
    r_jk = xyz[angle_list[:, 2]] - xyz[angle_list[:, 1]]

    x = torch.sum(r_ji * r_jk, dim=-1)
    y = torch.cross(r_ji, r_jk)
    y = torch.norm(y, dim=-1)
    angle = torch.atan2(y, x)

    return angle


def m_idx_of_angles(angle_list,
                    nbr_list,
                    angle_start,
                    angle_end):
    """
    Get the array index of elements of an angle list.
    Args:
        angle_list (torch.LongTensor): directed indices
            of sets of three atoms that are all in each
            other's neighborhood.
        nbr_list (torch.LongTensor): directed indices
            of pairs of atoms that are in each other's
            neighborhood.
        angle_start (int): the first index in the angle
            list you want.
        angle_end (int): the last index in the angle list
            you want.
    Returns:
        idx (torch.LongTensor): `m` indices.
    Example:
        angle_list = torch.LongTensor([[0, 1, 2],
                                       [0, 1, 3]])
        nbr_list = torch.LongTensor([[0, 1],
                                    [0, 2],
                                    [0, 3],
                                    [1, 0],
                                    [1, 2],
                                    [1, 3],
                                    [2, 0],
                                    [2, 1],
                                    [2, 3],
                                    [3, 0],
                                    [3, 1],
                                    [3, 2]])

        # This means that message vectors m_ij are ordered
        # according to m = {m_01, m_01, m_03, m_10,
        # m_12, m_13, m_30, m_31, m_32}. Say we are interested
        # in indices 2 and 1 for each element in the angle list.
        # If we want to know what the corresponding indices
        # in m (or the nbr list) are, we would call `m_idx_of_angles`
        # with angle_start = 2, angle_end = 1 (if we want the
        # {2,1} and {3,1} indices), or angle_start = 1,
        # angle_end = 0 (if we want the {1,2} and {1,3} indices).
        # Say we choose angle_start = 2 and angle_end = 1. Then
        # we get the indices of {m_21, m_31}, which we can see
        # from the nbr list are [7, 10].


    """

    # expand nbr_list[:, 0] so it's repeated once
    # for every element of `angle_list`.
    repeated_nbr = nbr_list[:, 0].repeat(angle_list.shape[0], 1)
    reshaped_angle = angle_list[:, angle_start].reshape(-1, 1)
    # gives you a matrix that shows you where each angle is equal
    # to nbr_list[:, 0]
    mask = repeated_nbr == reshaped_angle

    # same idea, but with nbr_list[:, 1] and angle_list[:, angle_end]

    repeated_nbr = nbr_list[:, 1].repeat(angle_list.shape[0], 1)
    reshaped_angle = angle_list[:, angle_end].reshape(-1, 1)

    # the full mask is the product of both
    mask *= (repeated_nbr == reshaped_angle)

    # get the indices where everything is true
    idx = mask.nonzero()[:, 1]

    return idx


class DimeNet(nn.Module):

    """DimeNet implementation.
    Source code (Tensorflow): https://github.com/klicperajo/dimenet
    Pytorch implementation: https://github.com/akirasosa/pytorch-dimenet

    Attributes:
        radial_basis (nff.nn.RadialBasis): radial basis layers for
            distances.
        spherical_basis (nff.nn.SphericalBasis): spherical basis for
            both distances and angles.
        embedding_block (nff.nn.EmbeddingBlock): block to convert
            atomic numbers into embedding vectors and concatenate
            embeddings and distances to make message embeddings.
        interaction_blocks (nn.ModuleList[nff.nn.InteractionBlock]):
            blocks for aggregating distance and angle information
            from neighboring atoms.
        output_blocks (nn.ModuleDict): Module  dictionary. Each
            key of the dictionary corresponds to a different property
            prediction, and its value is of type nn.ModuleList[nff.nn.
            OutputBlock]. These output blocks aggregate information
            at each interaction block and add it to the final result.
        out_keys (list): list of properties to be predicted by the
            network.
        grad_keys (list): list of properties for which we want the
            gradient to be computed.

    """

    def __init__(self, modelparams):
        """

        To instantiate the model, provide a dictionary with the required
        keys. 
        Args:
            modelparams (dict): parameters for model
        Returns:
            None
        For example:

            modelparams = {"n_rbf": 6,
                           "cutoff": 5.0,
                           "envelope_p": 6,
                           "n_spher": 6,
                           "l_spher": 7,
                           "embed_dim": 128,
                           "n_bilinear": 8,
                           "activation": "swish",
                           "n_convolutions": 6,
                           "output_keys": ["energy"],
                           "grad_keys": ["energy_grad"]}
            dime_net = DimeNet(modelparams)

        """

        super().__init__()

        self.radial_basis = RadialBasis(
            n_rbf=modelparams["n_rbf"],
            cutoff=modelparams["cutoff"],
            envelope_p=modelparams["envelope_p"])

        self.spherical_basis = SphericalBasis(
            n_spher=modelparams["n_spher"],
            l_spher=modelparams["l_spher"],
            cutoff=modelparams["cutoff"],
            envelope_p=modelparams["envelope_p"])

        self.embedding_block = EmbeddingBlock(
            n_rbf=modelparams["n_rbf"],
            embed_dim=modelparams["embed_dim"],
            activation=modelparams["activation"])

        self.interaction_blocks = nn.ModuleList([
            InteractionBlock(embed_dim=modelparams["embed_dim"],
                             n_rbf=modelparams["n_rbf"],
                             activation=modelparams["activation"],
                             n_spher=modelparams["n_spher"],
                             l_spher=modelparams["l_spher"],
                             n_bilinear=modelparams["n_bilinear"])
            for _ in range(modelparams["n_convolutions"])
        ])

        self.output_blocks = nn.ModuleDict(
            {key: nn.ModuleList([
                OutputBlock(embed_dim=modelparams["embed_dim"],
                            n_rbf=modelparams["n_rbf"],
                            activation=modelparams["activation"])
                for _ in range(modelparams["n_convolutions"] + 1)
            ])
                for key in modelparams["output_keys"]})

        self.out_keys = modelparams["output_keys"]
        self.grad_keys = modelparams["grad_keys"]

    def get_prelims(self, batch):
        """
        Get some quantities that we'll need for model prediction.
        Args:
            batch (dict): batch dictionary
        Returns:
            xyz (torch.Tensor): atom coordinates
            e_rbf (torch.Tensor): edge features in radial basis
            a_sbf (torch.Tensor): angle and edge features in spherical
                basis
            nbr_list (torch.LongTensor): neigbor list
            angle_list (torch.LongTensor): angle list
            num_atoms (int): total number of atoms in this batch
            z (torch.LongTensor): atomic numbers of atoms
            kj_idx (torch.LongTensor): nbr_list indices corresponding
                to the k,j indices in the angle list.
            ji_idx (torch.LongTensor): nbr_list indices corresponding
                to the j,i indices in the angle list.
        """

        nbr_list = batch["nbr_list"]
        angle_list = batch["angle_list"]
        nxyz = batch["nxyz"]
        num_atoms = batch["num_atoms"].sum()
        ji_idx = batch["ji_idx"]
        kj_idx = batch["kj_idx"]

        xyz = nxyz[:, 1:]
        z = nxyz[:, 0].long()
        xyz.requires_grad = True


        # compute distances
        d = torch.norm(xyz[nbr_list[:, 0]] - xyz[nbr_list[:, 1]],
                       dim=-1).reshape(-1, 1)

        # compute angles
        alpha = compute_angle(xyz, angle_list)

        # put the distances in the radial basis
        e_rbf = self.radial_basis(d)

        # put the distances and angles in the spherical basis
        a_sbf = self.spherical_basis(d, alpha, kj_idx)

        return (xyz, e_rbf, a_sbf, nbr_list, angle_list, num_atoms,
                z, kj_idx, ji_idx)

    def atomwise(self, batch):
        """
        Get atomwise outputs for each quantity.
        Args:
            batch (dict): batch dictionary
        Returns:
            out (dict): dictionary of atomwise outputs
            xyz (torch.Tensor): atom coordinates
        """

        (xyz, e_rbf, a_sbf, nbr_list, angle_list,
         num_atoms, z, kj_idx, ji_idx) = self.get_prelims(batch)

        # embed edge vectors
        m_ji = self.embedding_block(e_rbf=e_rbf,
                                    z=z,
                                    nbr_list=nbr_list)

        # initialiez the output dictionary with the first
        # of the output blocks acting on m_ji
        out = {key: self.output_blocks[key][0](m_ji=m_ji,
                                               e_rbf=e_rbf,
                                               nbr_list=nbr_list,
                                               num_atoms=num_atoms)
               for key in self.out_keys}

        # cycle through the interaction blocks
        for i, int_block in enumerate(self.interaction_blocks):

            # update the edge vector
            m_ji = int_block(m_ji=m_ji,
                             e_rbf=e_rbf,
                             a_sbf=a_sbf,
                             kj_idx=kj_idx,
                             ji_idx=ji_idx)

            # add to the output by putting m_ji through
            # an output block
            for key in self.out_keys:
                out_block = self.output_blocks[key][i + 1]
                out[key] += out_block(m_ji=m_ji,
                                      e_rbf=e_rbf,
                                      nbr_list=nbr_list,
                                      num_atoms=num_atoms)

        return out, xyz

    def forward(self, batch):
        """
        Call the model
        Args:
            batch (dict): batch dictionary
        Returns:
            results (dict): dictionary of predictions
        """

        out, xyz = self.atomwise(batch)
        N = batch["num_atoms"].detach().cpu().tolist()
        results = {}

        for key, val in out.items():
            # split the outputs into those of each molecule
            split_val = torch.split(val, N)
            # sum the results for each molecule
            results[key] = torch.stack([i.sum() for i in split_val])

        # compute gradients

        for key in self.grad_keys:
            output = out[key.replace("_grad", "")]
            grad = compute_grad(output=output,
                                inputs=xyz)
            results[key] = grad

        return results
