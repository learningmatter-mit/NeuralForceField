"""

Functions for adding empirical dispersion on top of an NFF potential.
Note: atomic units used throughout

D3 dispersion data taken from PhysNet:
https://github.com/MMunibas/PhysNet/tree/master/neural_network/grimme_d3/tables
"""

import os
import numpy as np
import torch
import json

from nff.utils import constants as const
from nff.utils.scatter import scatter_add


base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'table_data')
c6_ref_path = os.path.join(base_dir, 'c6ab.npy')
r_cov_path = os.path.join(base_dir, 'rcov.npy')
r2r4_path = os.path.join(base_dir, 'r2r4.npy')
func_path = os.path.join(base_dir, "functional_params.json")

# reference C6 data for pairs of atom types AB in different reference systems
# (up to 5 reference systems per pair)
C6_REF = torch.Tensor(np.load(c6_ref_path))

# covalent radii already scaled by k2 = 4 / 3. We unscale it here just so
# we can use them for other values if k2 if we want
R_COV = torch.Tensor(np.load(r_cov_path)) * 3 / 4

# table of Q = s_42 * sqrt(Z) * (<r^4> / <r^2>), for each atom type Z. This
# gets used to compute C8 from c6
R2R4 = torch.Tensor(np.load(r2r4_path))

# load json file with parameters for different DFT functionals
with open(func_path, "r") as f:
    FUNC_PARAMS = json.load(f)


def get_nbrs(batch,
             xyz,
             nbrs=None,
             mol_idx=None):
    """
    Get the undirected neighbor list connecting every atom to its neighbor within
    a given geometry, but not to itself or to atoms in other geometries.
    """

    num_atoms = batch['num_atoms']
    if not isinstance(num_atoms, list):
        num_atoms = num_atoms.tolist()

    if nbrs is None:

        nxyz_list = torch.split(batch['nxyz'], num_atoms)
        counter = 0

        nbrs = []

        for nxyz in nxyz_list:
            n = nxyz.shape[0]
            idx = torch.arange(n)
            x, y = torch.meshgrid(idx, idx)

            # undirected neighbor list
            these_nbrs = torch.cat([x.reshape(-1, 1), y.reshape(-1, 1)], dim=1)
            these_nbrs = these_nbrs[these_nbrs[:, 0] != these_nbrs[:, 1]]

            nbrs.append(these_nbrs + counter)
            counter += n

        nbrs = torch.cat(nbrs).to(xyz.device)

    z = batch['nxyz'][:, 0].long().to(xyz.device)

    if mol_idx is None:
        mol_idx = torch.cat([torch.zeros(num) + i
                             for i, num in enumerate(num_atoms)]
                            ).long().to(xyz.device)

    return nbrs, mol_idx, z


def get_coordination(xyz,
                     z,
                     nbrs,
                     r_cov,
                     k1,
                     k2):
    """
    Get coordination numbers for each atom
    """

    # distances in Bohr radii (atomic units)
    r_ab = ((xyz[nbrs[:, 0]] - xyz[nbrs[:, 1]])
            .pow(2).sum(1).sqrt() / const.BOHR_RADIUS)
    ra_cov = r_cov[z[nbrs[:, 0]]].to(r_ab.device)
    rb_cov = r_cov[z[nbrs[:, 1]]].to(r_ab.device)

    cn_ab = 1 / (1 + torch.exp(-k1 * (k2 * (ra_cov + rb_cov) / r_ab - 1)))
    cn = scatter_add(cn_ab,
                     nbrs[:, 0],
                     dim_size=xyz.shape[0])

    return cn, r_ab


def get_c6(z,
           cn,
           nbrs,
           c6_ref,
           k3):
    """
    Get the C6 parameter for each atom pair
    """

    c6ab_ref = c6_ref[z[nbrs[:, 0]], z[nbrs[:, 1]]].to(cn.device)
    cn_a_i = cn[nbrs[:, 0]]
    cn_b_j = cn[nbrs[:, 1]]

    c6_ab_ref_ij = c6ab_ref[:, :, :, 0]
    cn_a = c6ab_ref[:, :, :, 1]
    cn_b = c6ab_ref[:, :, :, 2]

    r = ((cn_a - cn_a_i.reshape(-1, 1, 1)) ** 2 +
         (cn_b - cn_b_j.reshape(-1, 1, 1)) ** 2)
    l_ij = torch.zeros_like(r)

    # exclude any info that doesn't exist for this (i, j) combination --
    # signified in the tables by c6_ab_ref = -1

    valid_idx = torch.bitwise_and(torch.bitwise_and(cn_a >= 0, cn_b >= 0),
                                  c6_ab_ref_ij >= 0)
    l_ij[valid_idx] = torch.exp(-k3 * r[valid_idx])

    w = l_ij.sum((1, 2))
    z_term = (c6_ab_ref_ij * l_ij).sum((1, 2))
    c6 = z_term / w

    return c6


def get_c8(z,
           nbrs,
           c6,
           r2r4):
    """
    Get the C6 parameter for each atom pair
    """

    r2r4 = r2r4.to(z.device)
    c8 = 3 * c6 * (r2r4[z[nbrs[:, 0]]] * r2r4[z[nbrs[:, 1]]])
    return c8


def disp_from_data(r_ab,
                   c6,
                   c8,
                   s6,
                   s8,
                   a1,
                   a2,
                   xyz,
                   nbrs,
                   mol_idx):

    r0_ab = (c8 / c6) ** 0.5
    f = a1 * r0_ab + a2

    e_ab = -1 / 2 * (s6 * c6 / (r_ab ** 6 + f ** 6) +
                     s8 * c8 / (r_ab ** 8 + f ** 8))
    e_a = scatter_add(e_ab,
                      nbrs[:, 0],
                      dim_size=xyz.shape[0])

    e_disp = scatter_add(e_a,
                         mol_idx,
                         dim_size=int(1 + mol_idx.max()))

    return e_disp


def get_func_info(functional,
                  disp_type,
                  func_params):

    msg = "Parameters not present for dispersion type %s" % disp_type
    func_params = {key.lower(): val for key, val in func_params.items()}
    assert disp_type.lower() in func_params, msg

    msg = ("Parameters not present for functional %s with dispersion type %s"
           % (functional, disp_type))

    sub_params = {key.lower(): val for key, val in
                  func_params[disp_type.lower()].items()}
    assert functional.lower() in sub_params, msg

    all_params = sub_params[functional.lower()]
    all_params.update(sub_params['universal'])

    return all_params


def get_dispersion(batch,
                   xyz,
                   disp_type,
                   functional,
                   c6_ref=C6_REF,
                   r_cov=R_COV,
                   r2r4=R2R4,
                   func_params=FUNC_PARAMS,
                   nbrs=None,
                   mol_idx=None):

    params = get_func_info(functional=functional,
                           disp_type=disp_type,
                           func_params=func_params)

    nbrs, mol_idx, z = get_nbrs(batch=batch,
                                xyz=xyz,
                                nbrs=nbrs,
                                mol_idx=mol_idx)
    cn, r_ab = get_coordination(xyz=xyz,
                                z=z,
                                nbrs=nbrs,
                                r_cov=r_cov,
                                k1=params["k1"],
                                k2=params["k2"])
    c6 = get_c6(z=z,
                cn=cn,
                nbrs=nbrs,
                c6_ref=c6_ref,
                k3=params["k3"])

    c8 = get_c8(z=z,
                nbrs=nbrs,
                c6=c6,
                r2r4=r2r4)

    e_disp = disp_from_data(r_ab=r_ab,
                            c6=c6,
                            c8=c8,
                            s6=params["s6"],
                            s8=params["s8"],
                            a1=params["a1"],
                            a2=params["a2"],
                            xyz=xyz,
                            nbrs=nbrs,
                            mol_idx=mol_idx)

    return e_disp
