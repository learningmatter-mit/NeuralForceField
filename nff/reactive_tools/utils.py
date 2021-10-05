from ase.vibrations import Vibrations
from ase.units import Bohr, mol, kcal
from ase import Atoms

import numpy as np

from rdkit import Chem


PT = Chem.GetPeriodicTable()

HA2J = 4.359744E-18
BOHRS2ANG = 0.529177
SPEEDOFLIGHT = 2.99792458E8
AMU2KG = 1.660538782E-27


def neural_hessian_ase(ase_atoms):
    print("Calculating Numerical Hessian using ASE")
    vib = Vibrations(ase_atoms, delta=0.05)
    vib.run()
    vib.summary()
    hessian = np.array(vib.H) * (kcal/mol) * Bohr**2
    vib.clean()

    return hessian


def neural_energy_ase(ase_atoms):
    return ase_atoms.get_potential_energy()[0]


def neural_force_ase(ase_atoms):
    return ase_atoms.get_forces()


def xyz_to_ase_atoms(xyz_file):
    sym = []
    pos = []

    f = open(xyz_file, "r")
    lines = f.readlines()

    for i, line in enumerate(lines):
        if i > 1:
            element, x, y, z = line.split()
            sym.append(element)
            pos.append([float(x), float(y), float(z)])

    return Atoms(
        symbols=sym,
        positions=pos,
        pbc=False,
    )


def moi_tensor(massvec, expmassvec, xyz):
    # Center of Mass
    com = np.sum(expmassvec.reshape(-1, 3)
                 * xyz.reshape(-1, 3), axis=0) / np.sum(massvec)

    # xyz shifted to COM
    xyz_com = xyz.reshape(-1, 3) - com

    # Compute elements need to calculate MOI tensor
    mass_xyz_com_sq_sum = np.sum(
        expmassvec.reshape(-1, 3) * xyz_com ** 2, axis=0)

    mass_xy = np.sum(massvec * xyz_com[:, 0] * xyz_com[:, 1], axis=0)
    mass_yz = np.sum(massvec * xyz_com[:, 1] * xyz_com[:, 2], axis=0)
    mass_xz = np.sum(massvec * xyz_com[:, 0] * xyz_com[:, 2], axis=0)

    # MOI tensor
    moi = np.array([[mass_xyz_com_sq_sum[1] + mass_xyz_com_sq_sum[2], -1 * mass_xy, -1 * mass_xz],
                    [-1 * mass_xy, mass_xyz_com_sq_sum[0] +
                        mass_xyz_com_sq_sum[2], -1 * mass_yz],
                    [-1 * mass_xz, -1 * mass_yz, mass_xyz_com_sq_sum[0] + mass_xyz_com_sq_sum[1]]])

    # MOI eigenvectors and eigenvalues
    moi_eigval, moi_eigvec = np.linalg.eig(moi)

    return xyz_com, moi_eigvec


def trans_rot_vec(massvec, xyz_com, moi_eigvec):

    # Mass-weighted translational vectors
    zero_vec = np.zeros([len(massvec)])
    sqrtmassvec = np.sqrt(massvec)
    expsqrtmassvec = np.repeat(sqrtmassvec, 3)

    d1 = np.transpose(np.stack((sqrtmassvec, zero_vec, zero_vec))).reshape(-1)
    d2 = np.transpose(np.stack((zero_vec, sqrtmassvec, zero_vec))).reshape(-1)
    d3 = np.transpose(np.stack((zero_vec, zero_vec, sqrtmassvec))).reshape(-1)

    # Mass-weighted rotational vectors
    big_p = np.matmul(xyz_com, moi_eigvec)

    d4 = (np.repeat(big_p[:, 1], 3).reshape(-1)
          * np.tile(moi_eigvec[:, 2], len(massvec)).reshape(-1)
          - np.repeat(big_p[:, 2], 3).reshape(-1)
          * np.tile(moi_eigvec[:, 1], len(massvec)).reshape(-1)) * expsqrtmassvec

    d5 = (np.repeat(big_p[:, 2], 3).reshape(-1)
          * np.tile(moi_eigvec[:, 0], len(massvec)).reshape(-1)
          - np.repeat(big_p[:, 0], 3).reshape(-1)
          * np.tile(moi_eigvec[:, 2], len(massvec)).reshape(-1)) * expsqrtmassvec

    d6 = (np.repeat(big_p[:, 0], 3).reshape(-1)
          * np.tile(moi_eigvec[:, 1], len(massvec)).reshape(-1)
          - np.repeat(big_p[:, 1], 3).reshape(-1)
          * np.tile(moi_eigvec[:, 0], len(massvec)).reshape(-1)) * expsqrtmassvec

    d1_norm = d1 / np.linalg.norm(d1)
    d2_norm = d2 / np.linalg.norm(d2)
    d3_norm = d3 / np.linalg.norm(d3)
    d4_norm = d4 / np.linalg.norm(d4)
    d5_norm = d5 / np.linalg.norm(d5)
    d6_norm = d6 / np.linalg.norm(d6)

    dx_norms = np.stack((d1_norm,
                         d2_norm,
                         d3_norm,
                         d4_norm,
                         d5_norm,
                         d6_norm))

    return dx_norms


def vib_analy(r, xyz, hessian):

    # r is the proton number of atoms
    # xyz is the cartesian coordinates in Angstrom
    # Hessian elements in atomic units (Ha/bohr^2)

    massvec = np.array([PT.GetAtomicWeight(i.item()) * AMU2KG
                        for i in list(np.array(r.reshape(-1)).astype(int))])
    expmassvec = np.repeat(massvec, 3)
    sqrtinvmassvec = np.divide(1.0, np.sqrt(expmassvec))
    hessian_mwc = np.einsum('i,ij,j->ij', sqrtinvmassvec,
                            hessian, sqrtinvmassvec)
    hessian_eigval, hessian_eigvec = np.linalg.eig(hessian_mwc)

    xyz_com, moi_eigvec = moi_tensor(massvec, expmassvec, xyz)
    dx_norms = trans_rot_vec(massvec, xyz_com, moi_eigvec)

    P = np.identity(3 * len(massvec))
    for dx_norm in dx_norms:
        P -= np.outer(dx_norm, dx_norm)

    # Projecting the T and R modes out of the hessian
    mwhess_proj = np.dot(P.T, hessian_mwc).dot(P)

    hessian_eigval, hessian_eigvec = np.linalg.eigh(mwhess_proj)

    neg_ele = []
    for i, eigval in enumerate(hessian_eigval):
        if eigval < 0:
            neg_ele.append(i)

    hessian_eigval_abs = np.abs(hessian_eigval)

    pre_vib_freq_cm_1 = np.sqrt(
        hessian_eigval_abs * HA2J * 10e19) / (SPEEDOFLIGHT * 2 * np.pi * BOHRS2ANG * 100)

    vib_freq_cm_1 = pre_vib_freq_cm_1.copy()

    for i in neg_ele:
        vib_freq_cm_1[i] = -1 * pre_vib_freq_cm_1[i]

    trans_rot_elms = []
    for i, freq in enumerate(vib_freq_cm_1):
        # Modes that are less than 1.0cm-1 is not a normal mode
        if np.abs(freq) < 1.0:
            trans_rot_elms.append(i)

    force_constants_J_m_2 = np.delete(
        hessian_eigval * HA2J * 1e20 / (BOHRS2ANG ** 2) * AMU2KG, trans_rot_elms)

    proj_vib_freq_cm_1 = np.delete(vib_freq_cm_1, trans_rot_elms)
    proj_hessian_eigvec = np.delete(hessian_eigvec.T, trans_rot_elms, 0)

    return force_constants_J_m_2, proj_vib_freq_cm_1, proj_hessian_eigvec
