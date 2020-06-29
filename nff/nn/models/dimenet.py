import torch
from torch import nn

from nff.nn.modules.dimenet import (EmbeddingBlock, InteractionBlock,
    OutputBlock, ResidualBlock)
from nff.nn.layers import DimeNetRadialBasis as RadialBasis
from nff.nn.layers import DimeNetSphericalBasis as SphericalBasis
from nff.utils.scatter import compute_grad


def compute_angle(xyz, angle_list):
    r_ij = xyz[angle_list[:, 0]] - xyz[angle_list[:, 1]]
    r_jk = xyz[angle_list[:, 1]] - xyz[angle_list[:, 2]]

    dot_prod = (r_ij * r_jk).sum(-1)
    cos_angle = dot_prod / (torch.norm(r_ij) *
                            torch.norm(r_jk))
    angle = torch.acos(cos_angle)

    return angle

def m_idx_of_angles(angle_list,
                    nbr_list,
                    angle_start,
                    angle_end):

    num_angles = angle_list.shape[0]
    cond_0 = (angle_list[:, angle_start] == nbr_list[:, 0].reshape(
        -1, 1).expand(-1, num_angles))
    cond_1 = (angle_list[:, angle_end] == nbr_list[:, 1].reshape(
        -1, 1).expand(-1, num_angles))
    mask = cond_0 * cond_1

    if angle_start < angle_end:
        idx = mask.nonzero()[:, 0]
    else:
        idx = mask.transpose(0, 1).nonzero()[:, 1]

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
                       "atom_embed_dim": 128,
                       "n_bilinear": 8,
                       "activation": "swish",
                       "n_convolutions": 6,
                       "output_keys": ["energy"],
                       "grad_keys": ["energy_grad"]}
        """

        super().__init__()

        self.radial_basis = RadialBasis(n_rbf=modelparams["n_rbf"],
                                        cutoff=modelparams["cutoff"],
                                        envelope_p=modelparams["envelope_p"])
        self.spherical_basis = SphericalBasis(n_spher=modelparams["n_spher"],
                                              l_spher=modelparams["l_spher"],
                                              cutoff=modelparams["cutoff"],
                                              envelope_p=modelparams["envelope_p"])
        self.embedding_block = EmbeddingBlock(n_rbf=modelparams["n_rbf"],
                                              atom_embed_dim=modelparams["atom_embed_dim"],
                                              activation=modelparams["activation"])

        self.interaction_blocks = nn.ModuleList([
            InteractionBlock(atom_embed_dim=modelparams["atom_embed_dim"],
                             n_rbf=modelparams["n_rbf"],
                             activation=modelparams["activation"],
                             n_spher=modelparams["n_spher"],
                             l_spher=modelparams["l_spher"],
                             n_bilinear=modelparams["n_bilinear"])
            for _ in range(modelparams["n_convolutions"])
        ])

        self.output_blocks = nn.ModuleDict(
            {key: nn.ModuleList([
                OutputBlock(atom_embed_dim=modelparams["atom_embed_dim"],
                            n_rbf=modelparams["n_rbf"],
                            activation=modelparams["activation"])
                for _ in range(modelparams["n_convolutions"] + 1)
            ])
                for key in modelparams["output_keys"]})

        self.residual_block = ResidualBlock(
            atom_embed_dim=modelparams["atom_embed_dim"],
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

        # do we really want kj or do we want jk?
        # need to check

        kj_idx = m_idx_of_angles(angle_list=angle_list,
                                 nbr_list=nbr_list,
                                 angle_start=2,
                                 angle_end=1)

        
        d = torch.norm(xyz[nbr_list[:, 0]] - xyz[nbr_list[:, 1]],
                       dim=-1).reshape(-1, 1)

        alpha = compute_angle(xyz, angle_list)

        e_rbf = self.radial_basis(d)
        a_sbf = self.spherical_basis(d, alpha, kj_idx)



        return (xyz, e_rbf, a_sbf, nbr_list, angle_list, num_atoms,
                z, kj_idx)

    def atomwise(self, batch):

        (xyz, e_rbf, a_sbf, nbr_list, angle_list,
         num_atoms, z, kj_idx) = self.get_prelims(batch)

        m_ji = self.embedding_block(e_rbf=e_rbf,
                                    z=z,
                                    nbr_list=nbr_list)

        out = {key: self.output_blocks[key][0](m_ji=m_ji,
                                               e_rbf=e_rbf,
                                               nbr_list=nbr_list,
                                               num_atoms=num_atoms)
               for key in self.out_keys}

        for int_block, out_block in zip(self.interaction_blocks,
                                        self.out_blocks[1:]):
            m_ji = int_block(m_ji=m_ji,
                             nbr_list=nbr_list,
                             angle_list=angle_list,
                             e_rbf=e_rbf,
                             a_sbf=a_sbf,
                             kj_idx=kj_idx)

            for key in self.out_keys:
                out[key] += out_block[key](m_ji=m_ji,
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
            grad = compute_grad(output=results[key], 
                inputs=xyz)
            results[key + "_grad"] = grad

        return results
