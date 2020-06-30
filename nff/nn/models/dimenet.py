import torch
from torch import nn

from nff.nn.modules.dimenet import (EmbeddingBlock, InteractionBlock,
                                    OutputBlock, ResidualBlock)
from nff.nn.layers import DimeNetRadialBasis as RadialBasis
from nff.nn.layers import DimeNetSphericalBasis as SphericalBasis
from nff.utils.scatter import compute_grad


def compute_angle(xyz, angle_list):

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

    repeated_nbr = nbr_list[:, 0].repeat(angle_list.shape[0], 1)
    reshaped_angle = angle_list[:, angle_start].reshape(-1, 1)
    mask = repeated_nbr == reshaped_angle

    repeated_nbr = nbr_list[:, 1].repeat(angle_list.shape[0], 1)
    reshaped_angle = angle_list[:, angle_end].reshape(-1, 1)
    mask *= (repeated_nbr == reshaped_angle)

    idx = mask.nonzero()[:, 1]

    return idx


class DimeNet(nn.Module):
    def __init__(self, modelparams):
        """
        Example: 

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

        self.residual_block = ResidualBlock(
            embed_dim=modelparams["embed_dim"],
            n_rbf=modelparams["n_rbf"],
            activation=modelparams["activation"])

        self.out_keys = modelparams["output_keys"]
        self.grad_keys = modelparams["grad_keys"]

    def get_prelims(self, batch):

        nbr_list = batch["nbr_list"]
        angle_list = batch["angle_list"]
        nxyz = batch["nxyz"]
        num_atoms = batch["num_atoms"].sum()

        xyz = nxyz[:, 1:]
        z = nxyz[:, 0].long()
        xyz.requires_grad = True

        # given an angle a_{ijk}, we want
        # ji_idx, which is the array index of m_ji.
        # We also want kj_idx, which is the array index
        # of m_kj. For example, if i,j,k = 0,1,2,
        # and our neighbor list is [[0, 1], [0, 2],
        # [1, 0], [1, 2], [2, 0], [2, 1]], then m_10 occurs
        # at index 2, and m_21 occurs at index 5. So 
        # ji_idx = 2 and kj_idx = 5.

        ji_idx = m_idx_of_angles(angle_list=angle_list,
                                 nbr_list=nbr_list,
                                 angle_start=1,
                                 angle_end=0)

        kj_idx = m_idx_of_angles(angle_list=angle_list,
                                 nbr_list=nbr_list,
                                 angle_start=2,
                                 angle_end=1)

        # Should we just have instantiated the m_ji
        # according to (a[:, 0], a[:, 1])? I think
        # that would make this much much easier.


        # do we also need an ij idx too?! I feel like we're 
        # using kj sometimes when we should be using ij.
        # For example, m_kj = self.m_kj_dense(m_ji[kj_idx])
        # seems wrong from this definition.

        d = torch.norm(xyz[nbr_list[:, 0]] - xyz[nbr_list[:, 1]],
                       dim=-1).reshape(-1, 1)

        alpha = compute_angle(xyz, angle_list)
        e_rbf = self.radial_basis(d)

        # is this right? Check if it should be kj_idx or ji_idx
        a_sbf = self.spherical_basis(d, alpha, kj_idx)

        return (xyz, e_rbf, a_sbf, nbr_list, angle_list, num_atoms,
                z, kj_idx, ji_idx)

    def atomwise(self, batch):

        (xyz, e_rbf, a_sbf, nbr_list, angle_list,
         num_atoms, z, kj_idx, ji_idx) = self.get_prelims(batch)

        m_ji = self.embedding_block(e_rbf=e_rbf,
                                    z=z,
                                    nbr_list=nbr_list)

        out = {key: self.output_blocks[key][0](m_ji=m_ji,
                                               e_rbf=e_rbf,
                                               nbr_list=nbr_list,
                                               num_atoms=num_atoms)
               for key in self.out_keys}

        for i, int_block in enumerate(self.interaction_blocks):

            m_ji = int_block(m_ji=m_ji,
                             nbr_list=nbr_list,
                             angle_list=angle_list,
                             e_rbf=e_rbf,
                             a_sbf=a_sbf,
                             kj_idx=kj_idx,
                             ji_idx=ji_idx)

            for key in self.out_keys:
                out_block = self.output_blocks[key][i + 1]
                out[key] += out_block(m_ji=m_ji,
                                      e_rbf=e_rbf,
                                      nbr_list=nbr_list,
                                      num_atoms=num_atoms)

        return out, xyz

    def forward(self, batch):

        out, xyz = self.atomwise(batch)
        N = batch["num_atoms"].detach().cpu().tolist()
        results = {}

        for key, val in out.items():
            split_val = torch.split(val, N)
            results[key] = torch.stack([i.sum() for i in split_val])

        for key in self.grad_keys:
            output = out[key.replace("_grad", "")]
            grad = compute_grad(output=output,
                                inputs=xyz)
            results[key] = grad

        return results
