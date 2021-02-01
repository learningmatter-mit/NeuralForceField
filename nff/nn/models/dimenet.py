import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import copy

from nff.nn.modules.dimenet import (EmbeddingBlock, InteractionBlock,
                                    OutputBlock)
from nff.nn.layers import DimeNetRadialBasis as RadialBasis
from nff.nn.layers import DimeNetSphericalBasis as SphericalBasis
from nff.nn.layers import Diagonalize
from nff.nn.graphop import batch_and_sum
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
                             n_bilinear=modelparams.get("n_bilinear"),
                             int_dim=modelparams.get("int_dim"),
                             basis_emb_dim=modelparams.get("basis_emb_dim"),
                             use_pp=modelparams.get("use_pp", False))
            for _ in range(modelparams["n_convolutions"])
        ])

        self.output_blocks = nn.ModuleDict(
            {key: nn.ModuleList([
                OutputBlock(embed_dim=modelparams["embed_dim"],
                            n_rbf=modelparams["n_rbf"],
                            activation=modelparams["activation"],
                            use_pp=modelparams.get("use_pp"),
                            out_dim=modelparams.get("out_dim"))
                for _ in range(modelparams["n_convolutions"] + 1)
            ])
                for key in modelparams["output_keys"]})

        self.out_keys = modelparams["output_keys"]
        self.grad_keys = modelparams["grad_keys"]

    def get_prelims(self, batch, xyz=None):
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

        z = nxyz[:, 0].long()
        if xyz is None:
            xyz = nxyz[:, 1:]
            if xyz.is_leaf:
                xyz.requires_grad = True

        ji_idx = batch["ji_idx"]
        kj_idx = batch["kj_idx"]

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

    def atomwise(self, batch, xyz=None):
        """
        Get atomwise outputs for each quantity.
        Args:
            batch (dict): batch dictionary
        Returns:
            out (dict): dictionary of atomwise outputs
            xyz (torch.Tensor): atom coordinates
        """

        (new_xyz, e_rbf, a_sbf, nbr_list, angle_list,
         num_atoms, z, kj_idx, ji_idx) = self.get_prelims(batch, xyz)

        if xyz is None:
            xyz = new_xyz

        # embed edge vectors
        m_ji = self.embedding_block(e_rbf=e_rbf,
                                    z=z,
                                    nbr_list=nbr_list)

        # initialize the output dictionary with the first
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

    def forward(self, batch, xyz=None):
        """
        Call the model
        Args:
            batch (dict): batch dictionary
        Returns:
            results (dict): dictionary of predictions
        """

        out, xyz = self.atomwise(batch, xyz)
        N = batch["num_atoms"].detach().cpu().tolist()
        results = {}

        for key, val in out.items():
            # split the outputs into those of each molecule
            split_val = torch.split(val, N)
            # sum the results for each molecule
            results[key] = torch.stack([i.sum() for i in split_val])

        # compute gradients

        for key in self.grad_keys:
            output = results[key.replace("_grad", "")]
            grad = compute_grad(output=output,
                                inputs=xyz)
            results[key] = grad

        return results


class DimeNetDiabat(DimeNet):

    def __init__(self, modelparams):
        """
        Diabat keys has the shape of a 2x2 matrix
        """

        energy_keys = modelparams["output_keys"]
        diabat_keys = modelparams["diabat_keys"]
        new_out_keys = list(set(np.array(diabat_keys).reshape(-1)
                                .tolist()))

        new_modelparams = copy.deepcopy(modelparams)
        new_modelparams.update({"output_keys": new_out_keys})
        super().__init__(new_modelparams)

        self.diag = Diagonalize()
        self.diabat_keys = diabat_keys
        self.grad_keys = modelparams["grad_keys"]
        self.energy_keys = energy_keys

    def add_grad(self, results, xyz):

        for grad_key in self.grad_keys:
            base_key = grad_key.replace("_grad", "")
            output = results[base_key]
            grad = compute_grad(inputs=xyz, output=output)

            results[grad_key] = grad

        return results

    def get_nacv(self, U, xyz, N):

        num_states = U.shape[2]
        split_xyz = torch.split(xyz, N)

        u_grads = [torch.zeros(num_states, num_states,
                               this_xyz.shape[0], 3).to(xyz.device)
                   for this_xyz in split_xyz]

        for i in range(U.shape[1]):
            for j in range(U.shape[2]):
                if i == j:
                    continue
                this_grad = compute_grad(inputs=xyz,
                                         output=U[:, i, j]).detach()
                grad_split = torch.split(this_grad, N)
                for k, grad in enumerate(grad_split):
                    u_grads[k][i, j, :, :] = grad

        U = U.detach()

        # m, l, and s are state indices that get summed out
        # i and j are state indices that don't get summed out
        # a = N_at is the number of atoms
        # t = 3 is the number of directions for each atom

        nacvs = []
        for k, u_grad in enumerate(u_grads):
            this_u = U[k]
            nacv = torch.einsum('im, sjat, lm, sl -> ijat',
                                this_u, u_grad, this_u, this_u)
            nacvs.append(nacv)

        # concatenate along all the atoms just like we do
        # for forces, giving a tensor of shape
        # (num_states, num_states, total_num_atoms, 3)

        nacv_cat = torch.cat(nacvs, axis=-2)

        return nacv_cat

    def add_nacv(self, results, U, xyz, N):

        nacv = self.get_nacv(U=U, xyz=xyz, N=N)
        num_states = nacv.shape[0]
        for i in range(num_states):
            for j in range(num_states):
                if i == j:
                    continue
                this_nacv = nacv[i, j, :, :]
                results[f"nacv_{i}{j}"] = this_nacv
        return results

    def add_diag(self, results, N, xyz, add_nacv):

        diabat_keys = np.array(self.diabat_keys)

        dim = diabat_keys.shape[0]
        num_geoms = len(N)
        diabat_ham = torch.zeros(num_geoms, dim, dim
                                 ).to(xyz.device)
        for i in range(dim):
            for j in range(dim):
                key = diabat_keys[i, j]
                diabat_ham[:, i, j] = results[key]

        ad_energies, U = self.diag(diabat_ham)

        results.update({key: ad_energies[:, i].reshape(-1, 1)
                        for i, key in enumerate(self.energy_keys)})
        if add_nacv:
            results = self.add_nacv(results=results,
                                    U=U,
                                    xyz=xyz,
                                    N=N)

        return results

    def add_gap(self, results):

        bottom_key = self.diabat_keys[0][0]
        top_key = self.diabat_keys[1][1]
        gap = results[top_key] - results[bottom_key]
        results.update({"abs_diabat_gap": abs(gap)})

        return results

    def forward(self,
                batch,
                xyz=None,
                add_nacv=False):

        out, xyz = self.atomwise(batch, xyz)
        N = batch["num_atoms"].detach().cpu().tolist()
        results = {}

        for key, val in out.items():
            # split the outputs into those of each molecule
            split_val = torch.split(val, N)
            # sum the results for each molecule
            results[key] = torch.stack([i.sum() for i in split_val])

        results = self.add_diag(results, N, xyz, add_nacv=add_nacv)
        results = self.add_grad(results, xyz)
        results = self.add_gap(results)

        return results


class DimeNetDiabatDelta(DimeNetDiabat):

    def __init__(self, modelparams):
        super().__init__(modelparams)

    def forward(self,
                batch,
                xyz=None,
                add_nacv=False):

        out, xyz = self.atomwise(batch, xyz)
        N = batch["num_atoms"].detach().cpu().tolist()
        results = {}

        for key, val in out.items():
            # split the outputs into those of each molecule
            split_val = torch.split(val, N)
            # sum the results for each molecule
            results[key] = torch.stack([i.sum() for i in split_val])

        diag_diabat_keys = np.diag(np.array(self.diabat_keys))
        diabat_0 = diag_diabat_keys[0]

        for key in diag_diabat_keys[1:]:
            results[key] += results[diabat_0]

        results = self.add_diag(results, N, xyz, add_nacv=add_nacv)
        results = self.add_grad(results, xyz)
        results = self.add_gap(results)

        return results


class DimeNetDelta(DimeNet):

    def __init__(self, modelparams):
        super().__init__(modelparams)

    def forward(self, batch, xyz=None):
        out, xyz = self.atomwise(batch, xyz)
        N = batch["num_atoms"].detach().cpu().tolist()
        results = {}

        for key, val in out.items():
            # split the outputs into those of each molecule
            split_val = torch.split(val, N)
            # sum the results for each molecule
            results[key] = torch.stack([i.sum() for i in split_val])

        for key in self.out_keys[1:]:
            results[key] += results[self.out_keys[0]]

        # compute gradients

        for key in self.grad_keys:
            output = results[key.replace("_grad", "")]
            grad = compute_grad(output=output,
                                inputs=xyz)
            results[key] = grad

        return results
