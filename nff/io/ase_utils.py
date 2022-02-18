import numpy as np

from ase.constraints import FixConstraint
from ase.geometry import get_dihedrals_derivatives, get_angles_derivatives


def get_dihed_derivs(atoms,
                     idx):

    pos = atoms.get_positions()

    a0s = pos[idx[:, 0]]
    a1s = pos[idx[:, 1]]
    a2s = pos[idx[:, 2]]
    a3s = pos[idx[:, 3]]

    # vectors 0->1, 1->2, 2->3
    v0 = a1s - a0s
    v1 = a2s - a1s
    v2 = a3s - a2s

    derivs = get_dihedrals_derivatives(v0, v1, v2,
                                       cell=atoms.cell,
                                       pbc=atoms.pbc)

    return derivs


def get_dihed_forces(atoms,
                     idx,
                     k,
                     dihed_0):
    """
    `dihed_0` in radians
    """

    # TODO: we should catch anything close to 0 like in xTB
    # and set it to 1e-8

    dihed_derivs = np.radians(get_dihed_derivs(atoms=atoms,
                                               idx=idx))

    diheds = np.radians(atoms.get_dihedrals(idx))

    const = k.reshape(-1, 1, 1)

    # constraining potential has the form  -k * cos(phi - phi0)
    forces = -const * np.sin(diheds - dihed_0).reshape(-1, 1, 1) * dihed_derivs
    total_forces = np.zeros_like(atoms.get_positions())

    for these_idx, these_forces in zip(idx, forces):
        total_forces[these_idx] += these_forces

    return total_forces


def get_angle_derivs(atoms,
                     idx):

    pos = atoms.get_positions()

    a1s = pos[idx[:, 0]]
    a2s = pos[idx[:, 1]]
    a3s = pos[idx[:, 2]]

    v12 = a1s - a2s
    v32 = a3s - a2s

    derivs = get_angles_derivatives(v12, v32,
                                    cell=atoms.cell,
                                    pbc=atoms.pbc)

    return derivs

    return derivs


def get_angle_forces(atoms,
                     idx,
                     k,
                     angle_0):
    """
    `angle_0` in radians
    """

    angle_derivs = np.radians(get_angle_derivs(atoms=atoms,
                                               idx=idx))
    angles = np.radians(atoms.get_angles(idx))
    const = k.reshape(-1, 1, 1)

    # k (theta - theta_0) ** 2, not 1/2 k
    forces = -2 * const * (angles - angle_0).reshape(-1, 1, 1) * angle_derivs
    total_forces = np.zeros_like(atoms.get_positions())

    for these_idx, these_forces in zip(idx, forces):
        total_forces[these_idx] += these_forces

    return total_forces


class ConstrainAngles(FixConstraint):

    def __init__(self,
                 idx,
                 atoms,
                 force_consts,
                 targ_angles=None):

        self.idx = np.asarray(idx)

        if targ_angles is not None:
            self.targ_angles = np.radians(targ_angles).astype('float')
        else:
            self.targ_angles = np.radians(atoms.get_angles(self.idx))

        if (isinstance(force_consts, float) or isinstance(force_consts, int)):
            self.force_consts = np.array([float(force_consts)] * len(self.idx))
        else:
            assert len(force_consts) == len(self.idx)
            self.force_consts = force_consts

    def get_removed_dof(self, atoms):
        # no degrees of freedom are being fixed, they're just being constrained.
        # So return 0 just like in the Hookean class
        return 0

    def adjust_positions(self, atoms, new):
        return

    def adjust_forces(self, atoms, forces):
        new_forces = get_angle_forces(atoms=atoms,
                                      idx=self.idx,
                                      k=self.force_consts,
                                      angle_0=self.targ_angles)

        forces += new_forces

        return new_forces, forces


class ConstrainDihedrals(FixConstraint):

    def __init__(self,
                 idx,
                 atoms,
                 force_consts,
                 targ_diheds=None):

        self.idx = np.asarray(idx)

        if targ_diheds is not None:
            self.targ_diheds = np.radians(targ_diheds).astype('float')
            # use the same convention as ASE for dihedrals - anything less than 0
            # gets 2 Pi added to it
            self.targ_diheds[self.targ_diheds < 0] += 2 * np.pi
        else:
            self.targ_diheds = np.radians(atoms.get_dihedrals(self.idx))

        if (isinstance(force_consts, float) or isinstance(force_consts, int)):
            self.force_consts = np.array([float(force_consts)] * len(self.idx))
        else:
            assert len(force_consts) == len(self.idx)
            self.force_consts = force_consts

    def get_removed_dof(self, atoms):
        # no degrees of freedom are being fixed, they're just being constrained
        # So return 0 just like in the Hookean class
        return 0

    def adjust_positions(self, atoms, new):
        return

    def adjust_forces(self, atoms, forces):
        new_forces = get_dihed_forces(atoms=atoms,
                                      idx=self.idx,
                                      k=self.force_consts,
                                      dihed_0=self.targ_diheds)

        forces += new_forces

        return new_forces, forces
