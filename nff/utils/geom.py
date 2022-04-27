"""
Tools for analyzing and comparing geometries
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from nff.utils.scatter import scatter_add


BATCH_SIZE = 3000


def quaternion_to_matrix(q):

    q0 = q[:, 0]
    q1 = q[:, 1]
    q2 = q[:, 2]
    q3 = q[:, 3]

    R_q = torch.stack([q0**2 + q1**2 - q2**2 - q3**2,
                       2 * (q1 * q2 - q0 * q3),
                       2 * (q1 * q3 + q0 * q2),
                       2 * (q1 * q2 + q0 * q3),
                       q0**2 - q1**2 + q2**2 - q3**2,
                       2 * (q2 * q3 - q0 * q1),
                       2 * (q1 * q3 - q0 * q2),
                       2 * (q2 * q3 + q0 * q1),
                       q0**2 - q1**2 - q2**2 + q3**2]
                      ).transpose(0, 1).reshape(-1, 3, 3)

    return R_q


def rotation_matrix_from_points(m0,
                                m1,
                                store_grad=False):

    v0 = m0[:, None, :, :]
    # don't have to clone this because we don't modify its actual value below
    v1 = m1

    out_0 = (v0 * v1).sum(-1).reshape(-1, 3)
    r_11 = out_0[:, 0]
    r_22 = out_0[:, 1]
    r_33 = out_0[:, 2]

    out_1 = torch.sum(v0 * torch.roll(v1, -1, dims=1), dim=-1
                      ).reshape(-1, 3)
    r_12 = out_1[:, 0]
    r_23 = out_1[:, 1]
    r_31 = out_1[:, 2]

    out_2 = torch.sum(v0 * torch.roll(v1, -2, dims=1), dim=-1
                      ).reshape(-1, 3)
    r_13 = out_2[:, 0]
    r_21 = out_2[:, 1]
    r_32 = out_2[:, 2]

    f = torch.stack([r_11 + r_22 + r_33, r_23 - r_32, r_31 - r_13, r_12 - r_21,
                     r_23 - r_32, r_11 - r_22 - r_33, r_12 + r_21, r_13 + r_31,
                     r_31 - r_13, r_12 + r_21, -r_11 + r_22 - r_33, r_23 + r_32,
                     r_12 - r_21, r_13 + r_31, r_23 + r_32, -r_11 - r_22 + r_33]
                    ).transpose(0, 1).reshape(-1, 4, 4)

    # Really slow on a GPU / with torch for some reason.
    # See https://github.com/pytorch/pytorch/issues/22573:
    # the slow-down is significant in PyTorch, and is particularly
    # bad for small matrices.

    # Use numpy on cpu instead

    if store_grad:
        w, V = torch.linalg.eigh(f)
        arg = w.argmax(dim=1)
        idx = list(range(len(arg)))
        q = V[idx, :, arg]

        R = quaternion_to_matrix(q)

        return R

    f_np = f.detach().cpu().numpy()
    nan_idx = np.isnan(f_np).any(-1).any(-1)
    good_idx = np.bitwise_not(nan_idx)
    f_good = f_np[good_idx]

    if f_good.shape[0] != 0:
        # Only do this if we have any good idx
        # Otherwise we'll run into issues with
        # taking the argmax of an empty
        # sequence

        w, V = np.linalg.eigh(f_good)
        w = torch.Tensor(w).to(f.device)
        V = torch.Tensor(V).to(f.device)

        arg = w.argmax(dim=1)
        idx = list(range(len(arg)))
        q = V[idx, :, arg]

        R = quaternion_to_matrix(q)

    counter = 0
    r_with_nan = []

    for i, is_nan in enumerate(nan_idx):
        if is_nan:
            r_with_nan.append(torch.diag(torch.ones(3)))
            counter += 1
        else:
            r_with_nan.append(R[i - counter])
    r_with_nan = torch.stack(r_with_nan)

    return r_with_nan


def minimize_rotation_and_translation(targ_nxyz,
                                      this_nxyz,
                                      store_grad=False):

    base_p = this_nxyz[:, :, 1:]
    if store_grad:
        base_p.requires_grad = True
    p0 = targ_nxyz[:, :, 1:]

    c = base_p.mean(1).reshape(-1, 1, 3)
    p = base_p - c

    c0 = p0.mean(1).reshape(-1, 1, 3)
    p0 -= c0

    R = rotation_matrix_from_points(p.transpose(1, 2),
                                    p0.transpose(1, 2),
                                    store_grad=store_grad)

    num_repeats = targ_nxyz.shape[0]
    p_repeat = torch.repeat_interleave(p, num_repeats, dim=0)

    new_p = torch.einsum("ijk,ilk->ijl", p_repeat, R)

    return new_p, p0, R, base_p


def compute_rmsd(targ_nxyz,
                 this_nxyz):

    targ_nxyz = torch.Tensor(targ_nxyz).reshape(1, -1, 4)
    this_nxyz = torch.Tensor(this_nxyz).reshape(1, -1, 4)

    out = minimize_rotation_and_translation(targ_nxyz=targ_nxyz,
                                            this_nxyz=this_nxyz)
    xyz_0, new_targ, _, _ = out

    num_mols_1 = targ_nxyz.shape[0]
    num_mols_0 = this_nxyz.shape[0]

    xyz_1 = new_targ.repeat(num_mols_0, 1, 1)

    delta_sq = (xyz_0 - xyz_1) ** 2

    num_atoms = delta_sq.shape[1]
    distances = (((delta_sq.sum((1, 2)) / num_atoms) ** 0.5)
                 .reshape(num_mols_0, num_mols_1)
                 .cpu().reshape(-1).item())

    return distances


def compute_distance(targ_nxyz,
                     atom_nxyz,
                     store_grad=False):

    out = minimize_rotation_and_translation(targ_nxyz=targ_nxyz,
                                            this_nxyz=atom_nxyz,
                                            store_grad=store_grad)

    xyz_0, new_targ, R, base_p = out

    num_mols_1 = targ_nxyz.shape[0]
    num_mols_0 = atom_nxyz.shape[0]

    xyz_1 = new_targ.repeat(num_mols_0, 1, 1)

    delta_sq = (xyz_0 - xyz_1) ** 2

    num_atoms = delta_sq.shape[1]
    distances = ((delta_sq.sum((1, 2)) / num_atoms) **
                 0.5).reshape(num_mols_0, num_mols_1).cpu()
    R = R.cpu()

    if store_grad:
        return distances, R, base_p
    else:
        return distances.detach(), R


def compute_distances(dataset,
                      device,
                      batch_size=BATCH_SIZE,
                      dataset_1=None,
                      store_grad=False,
                      collate_dicts=None):
    """
    Compute distances between different configurations for one molecule.
    """

    if collate_dicts is None:
        from nff.data import collate_dicts

    if dataset_1 is None:
        dataset_1 = dataset

    distance_mat = torch.zeros((len(dataset), len(dataset_1)))
    shape = list(distance_mat.shape)
    shape += [3, 3]
    R_mat = torch.zeros(tuple(shape))

    loader_0 = DataLoader(dataset,
                          batch_size=batch_size,
                          collate_fn=collate_dicts)

    loader_1 = DataLoader(dataset_1,
                          batch_size=batch_size,
                          collate_fn=collate_dicts)

    i_start = 0
    xyz_list = []

    for batch_0 in loader_0:

        j_start = 0
        for batch_1 in loader_1:

            num_mols_0 = len(batch_0["num_atoms"])
            num_mols_1 = len(batch_1["num_atoms"])

            targ_nxyz = (batch_0["nxyz"]
                         .reshape(num_mols_0, -1, 4).to(device))
            atom_nxyz = (batch_1["nxyz"]
                         .reshape(num_mols_1, -1, 4).to(device))

            out = compute_distance(targ_nxyz=targ_nxyz,
                                   atom_nxyz=atom_nxyz,
                                   store_grad=store_grad)
            if store_grad:
                distances, R, xyz_0 = out
                num_atoms = batch_1["num_atoms"].tolist()
                xyz_list.append(xyz_0)

            else:
                distances, R = out

            distances = distances.transpose(0, 1)

            all_indices = (torch.ones_like(distances)
                           .nonzero(as_tuple=False)
                           .cpu())

            all_indices[:, 0] += i_start
            all_indices[:, 1] += j_start

            distance_mat[all_indices[:, 0],
                         all_indices[:, 1]] = distances.reshape(-1)

            R_mat[all_indices[:, 0],
                  all_indices[:, 1]] = R.detach()

            j_start += num_mols_1

        i_start += num_mols_0

    if store_grad:
        return distance_mat, R_mat, xyz_list
    else:
        return distance_mat, R_mat


"""
Below are a set of functions for batched coordinates from several molecules. They
all assume that you have one set of reference coordinates (i.e., one geometry
from each of the molecules, concatenated together), and multiple sets of query
coordinates  (i.e., several batched coordinates, all stacked). The use is primarily
for batched RMSD-based metadynamics, where a set of reference structures is compared
to the current coordinates.
"""


def batched_translate(ref_xyz,
                      query_xyz,
                      mol_idx,
                      num_atoms_tensor):
    """
    Translate a set of batched atomic coordinates concatenated together
    from different molecules, so they align with the COM of the reference molecule.

    Args:
        this_xyz (torch.Tensor): n_atoms x 3 current coordinates, where n_atoms is
            the sum of the number of atoms in all molecules
        query_xyz (torch.Tensor): n_samples x n_atoms x 3, where n_samples is the
            number of different query configurations (e.g. different saved structures
            from metadynamics)
        mol_idx (torch.LongTensor): tensor of dimension n_atom, where each item tells
            tells you the molecule index at the associated tensor index (e.g. [0, 0, 1]
            tells you that the first two atoms belong to molecule 0, and the third to
            molecule 1).
        num_atoms_tensor (torch.LongTensor): tensor of number of atoms in each molecule
    """

    ref_sum = scatter_add(src=ref_xyz,
                          index=mol_idx,
                          dim=0,
                          dim_size=mol_idx.max() + 1)
    ref_com = ref_sum / num_atoms_tensor.reshape(-1, 1)

    query_sum = scatter_add(src=query_xyz,
                            index=mol_idx,
                            dim=1,
                            dim_size=mol_idx.max() + 1)
    query_com = query_sum / num_atoms_tensor.reshape(-1, 1)

    ref_centered = ref_xyz - torch.repeat_interleave(ref_com,
                                                     num_atoms_tensor,
                                                     dim=0)
    # reshape to match query
    ref_centered = ref_centered.unsqueeze(0)

    query_centered = query_xyz - torch.repeat_interleave(query_com,
                                                         num_atoms_tensor,
                                                         dim=1)

    return ref_centered, query_centered


def rmat_from_batched_points(ref_centered,
                             query_centered,
                             mol_idx,
                             num_atoms_tensor,
                             store_grad=False):
    """
    Rotation matrix from a set of atomic coordinates concatenated together
    from different molecules.
    """

    out_0 = scatter_add(src=(ref_centered * query_centered),
                        index=mol_idx,
                        dim=1)

    r_11 = out_0[:, :, 0]
    r_22 = out_0[:, :, 1]
    r_33 = out_0[:, :, 2]

    out_1 = scatter_add(src=(ref_centered * torch.roll(query_centered, -1, dims=2)),
                        index=mol_idx,
                        dim=1)

    r_12 = out_1[:, :, 0]
    r_23 = out_1[:, :, 1]
    r_31 = out_1[:, :, 2]

    out_2 = scatter_add(src=(ref_centered * torch.roll(query_centered, -2, dims=2)),
                        index=mol_idx,
                        dim=1)

    r_13 = out_2[:, :, 0]
    r_21 = out_2[:, :, 1]
    r_32 = out_2[:, :, 2]

    f_0 = [r_11 + r_22 + r_33, r_23 - r_32, r_31 - r_13, r_12 - r_21]
    f_1 = [r_23 - r_32, r_11 - r_22 - r_33, r_12 + r_21, r_13 + r_31]
    f_2 = [r_31 - r_13, r_12 + r_21, - r_11 + r_22 - r_33, r_23 + r_32]
    f_3 = [r_12 - r_21, r_13 + r_31, r_23 + r_32, -r_11 - r_22 + r_33]

    f = torch.stack(
        [torch.stack(f_0),
         torch.stack(f_1),
         torch.stack(f_2),
         torch.stack(f_3)]
    ).permute(2, 3, 0, 1).reshape(-1, 4, 4)

    if store_grad:
        w, V = torch.linalg.eigh(f)
        arg = w.argmax(dim=1)
        idx = list(range(len(arg)))
        q = V[idx, :, arg]

        r = quaternion_to_matrix(q)
        # reshape it to differentiate between molecules in a batch and different
        # batches
        r = r.reshape(-1, num_atoms_tensor.shape[0], 3, 3)

        # repeat for each atom so we can do the matrix multiplication more easily
        # later
        r_repeat = torch.repeat_interleave(r, num_atoms_tensor, dim=1)

        return r_repeat

    raise NotImplementedError("Not yet implemented in numpy")


def batch_minimize_rot_trans(ref_nxyz,
                             query_nxyz,
                             mol_idx,
                             num_atoms_tensor,
                             store_grad=False):

    ref_xyz = ref_nxyz[:, 1:]
    if store_grad:
        ref_xyz.requires_grad = True
    query_xyz = query_nxyz[:, :, 1:]

    ref_centered, query_centered = batched_translate(ref_xyz=ref_xyz,
                                                     query_xyz=query_xyz,
                                                     mol_idx=mol_idx,
                                                     num_atoms_tensor=num_atoms_tensor)

    r = rmat_from_batched_points(ref_centered=ref_centered,
                                 query_centered=query_centered,
                                 mol_idx=mol_idx,
                                 num_atoms_tensor=num_atoms_tensor,
                                 store_grad=store_grad)

    query_center_rot = torch.einsum('...kj,...k->...j', r, query_centered)

    return ref_xyz, ref_centered, query_center_rot, r


def batch_compute_distance(ref_nxyz,
                           query_nxyz,
                           mol_idx,
                           num_atoms_tensor,
                           store_grad=False):

    out = batch_minimize_rot_trans(ref_nxyz=ref_nxyz,
                                   query_nxyz=query_nxyz,
                                   mol_idx=mol_idx,
                                   num_atoms_tensor=num_atoms_tensor,
                                   store_grad=store_grad)
    ref_xyz, ref_centered, query_center_rot, r = out
    delta_sq = (ref_centered - query_center_rot) ** 2
    delta_sq_sum = scatter_add(src=delta_sq,
                               index=mol_idx,
                               dim=1,
                               dim_size=mol_idx.max() + 1).sum(-1)

    delta_sq_mean = delta_sq_sum / num_atoms_tensor.reshape(1, -1)
    rmsd = delta_sq_mean ** 0.5

    return rmsd, ref_xyz
