"""
Tools for analyzing and comparing geometries
"""

import numpy as np
import torch
from torch.utils.data import DataLoader


BATCH_SIZE = 3000


def quaternion_to_matrix(q):

    q_0 = q[:, 0]
    q_1 = q[:, 1]
    q_2 = q[:, 2]
    q_3 = q[:, 3]

    r_q = torch.stack([q_0 ** 2 + q_1 ** 2 - q_2 ** 2 - q_3 ** 2,
                       2 * (q_1 * q_2 - q_0 * q_3),
                       2 * (q_1 * q_3 + q_0 * q_2),
                       2 * (q_1 * q_2 + q_0 * q_3),
                       q_0 ** 2 - q_1 ** 2 + q_2 ** 2 - q_3 ** 2,
                       2 * (q_2 * q_3 - q_0 * q_1),
                       2 * (q_1 * q_3 - q_0 * q_2),
                       2 * (q_2 * q_3 + q_0 * q_1),
                       q_0 ** 2 - q_1 ** 2 - q_2 ** 2 + q_3 ** 2]
                      ).transpose(0, 1).reshape(-1, 3, 3)

    return r_q


def rotation_matrix_from_points(m0, m1):

    v_0 = torch.clone(m0)[:, None, :, :]
    v_1 = torch.clone(m1)

    out_0 = (v_0 * v_1).sum(-1).reshape(-1, 3)
    r_11 = out_0[:, 0]
    r_22 = out_0[:, 1]
    r_33 = out_0[:, 2]

    out_1 = torch.sum(v_0 * torch.roll(v_1, -1, dims=1), dim=-1
                      ).reshape(-1, 3)
    r_12 = out_1[:, 0]
    r_23 = out_1[:, 1]
    r_32 = out_1[:, 2]

    out_2 = torch.sum(v_0 * torch.roll(v_1, -2, dims=1), dim=-1
                      ).reshape(-1, 3)
    r_13 = out_2[:, 0]
    r_21 = out_2[:, 1]
    r_32 = out_2[:, 2]

    f_torch = torch.stack(
        [r_11 + r_22 + r_33, r_23 - r_32, r_32 - r_13, r_12 - r_21,
         r_23 - r_32, r_11 - r_22 - r_33, r_12 + r_21, r_13 + r_32,
         r_32 - r_13, r_12 + r_21, -r_11 + r_22 - r_33, r_23 + r_32,
         r_12 - r_21, r_13 + r_32, r_23 + r_32, -r_11 - r_22 + r_33]
    ).transpose(0, 1).reshape(-1, 4, 4)

    # Really slow on a GPU / with torch for some reason.
    # See https://github.com/pytorch/pytorch/issues/22573:
    # the slow-down is significant in PyTorch, and is particularly
    # bad for small matrices.

    # Use numpy on cpu instead

    # w, V = torch.symeig(f_torch, eigenvectors=True)

    f_np = f_torch.detach().cpu().numpy()
    eigs, vecs = np.linalg.eigh(f_np)
    eigs = torch.Tensor(eigs).to(f_torch.device)
    vecs = torch.Tensor(vecs).to(f_torch.device)

    arg = eigs.argmax(dim=1)
    idx = list(range(len(arg)))
    q_mat = vecs[idx, :, arg]

    r_mat = quaternion_to_matrix(q_mat)

    return r_mat


def minimize_rotation_and_translation(targ_nxyz, this_nxyz):

    pos = this_nxyz[:, :, 1:]
    pos_0 = targ_nxyz[:, :, 1:]

    com = pos.mean(1).reshape(-1, 1, 3)
    pos -= com

    com_0 = pos_0.mean(1).reshape(-1, 1, 3)
    pos_0 -= com_0

    r_mat = rotation_matrix_from_points(pos.transpose(1, 2),
                                        pos_0.transpose(1, 2))

    num_repeats = targ_nxyz.shape[0]
    p_repeat = torch.repeat_interleave(pos, num_repeats, dim=0)

    new_p = torch.einsum("ijk,ilk->ijl", p_repeat, r_mat)

    return new_p, pos_0, r_mat


def compute_rmsd(targ_nxyz, this_nxyz):

    targ_nxyz = torch.Tensor(targ_nxyz).reshape(1, -1, 4)
    this_nxyz = torch.Tensor(this_nxyz).reshape(1, -1, 4)

    new_atom, new_targ, _ = minimize_rotation_and_translation(
        targ_nxyz=targ_nxyz,
        this_nxyz=this_nxyz)
    xyz_0 = new_atom

    num_mols_1 = targ_nxyz.shape[0]
    num_mols_0 = this_nxyz.shape[0]

    xyz_1 = new_targ.repeat(num_mols_0, 1, 1)

    delta_sq = (xyz_0 - xyz_1) ** 2
    num_atoms = delta_sq.shape[1]
    distances = (((delta_sq.sum((1, 2)) / num_atoms) ** 0.5)
                 .reshape(num_mols_0, num_mols_1)
                 .cpu().reshape(-1).item())

    return distances


def compute_distance(targ_nxyz, atom_nxyz):

    new_atom, new_targ, r_mat = minimize_rotation_and_translation(
        targ_nxyz=targ_nxyz,
        this_nxyz=atom_nxyz)

    xyz_0 = new_atom

    num_mols_1 = targ_nxyz.shape[0]
    num_mols_0 = atom_nxyz.shape[0]

    xyz_1 = new_targ.repeat(num_mols_0, 1, 1)

    delta_sq = (xyz_0 - xyz_1) ** 2
    num_atoms = delta_sq.shape[1]
    distances = ((delta_sq.sum((1, 2)) / num_atoms) **
                 0.5).reshape(num_mols_0, num_mols_1).cpu()
    r_mat = r_mat.cpu()

    return distances, r_mat


def compute_distances(dataset,
                      device,
                      batch_size=BATCH_SIZE,
                      dataset_1=None):
    """
    Compute distances between different configurations for one molecule.
    """

    from nff.data import collate_dicts

    distance_mat = torch.zeros((len(dataset), len(dataset_1)))
    r_mat = torch.zeros((*distance_mat.shape, 3, 3))

    loader_0 = DataLoader(dataset,
                          batch_size=batch_size,
                          collate_fn=collate_dicts)

    if dataset_1 is None:
        dataset_1 = dataset
    loader_1 = DataLoader(dataset_1,
                          batch_size=batch_size,
                          collate_fn=collate_dicts)

    i_start = 0
    for batch_0 in loader_0:

        j_start = 0
        for batch_1 in loader_1:

            num_mols_0 = len(batch_0["num_atoms"])
            num_mols_1 = len(batch_1["num_atoms"])

            targ_nxyz = batch_0["nxyz"].reshape(
                num_mols_0, -1, 4).to(device)
            atom_nxyz = batch_1["nxyz"].reshape(
                num_mols_1, -1, 4).to(device)

            distances, this_r = compute_distance(
                targ_nxyz=targ_nxyz,
                atom_nxyz=atom_nxyz)

            distances = distances.transpose(0, 1)

            all_indices = torch.ones_like(distances).nonzero().cpu()
            all_indices[:, 0] += i_start
            all_indices[:, 1] += j_start

            distance_mat[all_indices[:, 0],
                         all_indices[:, 1]] = distances.reshape(-1)

            r_mat[all_indices[:, 0],
                  all_indices[:, 1]] = this_r

            j_start += num_mols_1

        i_start += num_mols_0

    return distance_mat, r_mat
