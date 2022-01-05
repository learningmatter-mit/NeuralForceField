import torch
import numpy as np
from ase.constraints import FixInternals


def get_distances_derivatives(v0,
                              cell=None,
                              pbc=None):
    """Get derivatives of distances for all vectors in v0 w.r.t. Cartesian
    coordinates in Angstrom.

    Set cell and pbc to use the minimum image convention.

    There is a singularity for distances -> 0 for which a ZeroDivisionError is
    raised.
    Derivative output format: [[dx_a0, dy_a0, dz_a0], [dx_a1, dy_a1, dz_a1]].
    """

    if cell is not None:
        raise NotImplementedError("Not yet implemented for PBC")

    dists = torch.linalg.norm(v0, axis=-1)

    if (dists <= 0.).any():  # identify singularities
        raise ZeroDivisionError(('Singularity for derivative of a '
                                 'zero distance'))

    derivs_d0 = torch.einsum('i,ij->ij', -1. / dists,
                             v0)  # derivatives by atom 0
    derivs_d1 = -derivs_d0                              # derivatives by atom 1
    derivs = torch.stack((derivs_d0, derivs_d1), axis=1)
    return derivs


class TorchShakeBonds(FixInternals):
    """Constraint object for fixing multiple internal coordinates.

    Allows fixing bonds, angles, and dihedrals as well as linear combinations
    of bond lengths (bondcombos).
    Please provide angular units in degrees using angles_deg and
    dihedrals_deg.
    Fixing planar angles is not supported at the moment.
    """

    def __init__(self,
                 device,
                 *args,
                 **kwargs):

        super().__init__(*args, **kwargs)

        self.device = device
        for key, val in self.__dict__.items():
            if isinstance(val, np.ndarray):
                setattr(key, torch.Tensor(val).to(self.device))

    def get_removed_dof(self, atoms):
        return self.n

    def initialize(self, atoms):
        if self.initialized:
            return

        masses = torch.Tensor(atoms.get_masses()).to(self.device)
        masses = torch.repeat_interleave(masses, 3)
        cell = None
        pbc = None
        if self.mic:
            cell = atoms.cell
            pbc = atoms.pbc
        self.constraints = []
        for data, make_constr in [(self.bonds, self.FixBondLengthAlt),
                                  (self.angles, self.FixAngle),
                                  (self.dihedrals, self.FixDihedral),
                                  (self.bondcombos, self.FixBondCombo)]:
            for datum in data:
                constr = make_constr(self.device,
                                     datum[0], datum[1], masses, cell, pbc)
                self.constraints.append(constr)
        self.initialized = True

    def shuffle_definitions(self, shuffle_dic, internal_type):
        dfns = []  # definitions
        for dfn in internal_type:  # e.g. for bond in self.bonds
            append = True
            new_dfn = [dfn[0], list(dfn[1])]
            for old in dfn[1]:
                if old in shuffle_dic:
                    new_dfn[1][dfn[1].index(old)] = shuffle_dic[old]
                else:
                    append = False
                    break
            if append:
                dfns.append(new_dfn)
        return dfns

    def shuffle_combos(self, shuffle_dic, internal_type):
        dfns = []  # definitions
        for dfn in internal_type:  # e.g. for bondcombo in self.bondcombos
            append = True
            all_indices = [idx[0:-1] for idx in dfn[1]]
            new_dfn = [dfn[0], list(dfn[1])]
            for i, indices in enumerate(all_indices):
                for old in indices:
                    if old in shuffle_dic:
                        new_dfn[1][i][indices.index(old)] = shuffle_dic[old]
                    else:
                        append = False
                        break
                if not append:
                    break
            if append:
                dfns.append(new_dfn)
        return dfns

    def index_shuffle(self, atoms, ind):
        # See docstring of superclass
        self.initialize(atoms)
        shuffle_dic = dict(slice2enlist(ind, len(atoms)))
        shuffle_dic = {old: new for new, old in shuffle_dic.items()}
        self.bonds = self.shuffle_definitions(shuffle_dic, self.bonds)
        self.angles = self.shuffle_definitions(shuffle_dic, self.angles)
        self.dihedrals = self.shuffle_definitions(shuffle_dic, self.dihedrals)
        self.bondcombos = self.shuffle_combos(shuffle_dic, self.bondcombos)
        self.initialized = False
        self.initialize(atoms)
        if len(self.constraints) == 0:
            raise IndexError('Constraint not part of slice')

    def get_indices(self):
        cons = []
        for dfn in self.bonds + self.dihedrals + self.angles:
            cons.extend(dfn[1])
        for dfn in self.bondcombos:
            for partial_dfn in dfn[1]:
                cons.extend(partial_dfn[0:-1])  # last index is the coefficient
        return list(set(cons))

    def todict(self):
        return {'name': 'FixInternals',
                'kwargs': {'bonds': self.bonds,
                           'angles_deg': self.angles,
                           'dihedrals_deg': self.dihedrals,
                           'bondcombos': self.bondcombos,
                           'mic': self.mic,
                           'epsilon': self.epsilon}}

    def adjust_positions(self, atoms, newpos):
        self.initialize(atoms)

        oldpos = torch.Tensor(atoms.positions).to(self.device)
        torch_newpos = torch.Tensor(newpos).to(self.device)

        for constraint in self.constraints:
            constraint.setup_jacobian(oldpos)
        for j in range(50):
            maxerr = 0.0
            for constraint in self.constraints:
                constraint.adjust_positions(oldpos, torch_newpos)
                maxerr = max(abs(constraint.sigma), maxerr)
            if maxerr < self.epsilon:
                newpos[:] = torch_newpos.cpu().numpy()
                return
        msg = 'FixInternals.adjust_positions did not converge.'
        if any([constr.targetvalue > 175. or constr.targetvalue < 5. for constr
                in self.constraints if type(constr) is self.FixAngle]):
            msg += (' This may be caused by an almost planar angle.'
                    ' Support for planar angles would require the'
                    ' implementation of ghost, i.e. dummy, atoms.'
                    ' See issue #868.')
        raise ValueError(msg)

    def adjust_forces(self, atoms, forces):
        """Project out translations and rotations and all other constraints"""

        forces = torch.Tensor(forces).to(self.device)
        self.initialize(atoms)
        positions = torch.Tensor(atoms.positions).to(self.device)
        N = len(forces)
        list2_constraints = list(torch.zeros((6, N, 3)).to(self.device))
        tx, ty, tz, rx, ry, rz = list2_constraints

        list_constraints = [r.ravel() for r in list2_constraints]

        tx[:, 0] = 1.0
        ty[:, 1] = 1.0
        tz[:, 2] = 1.0
        ff = forces.ravel()

        # Calculate the center of mass
        center = positions.sum(axis=0) / N

        rx[:, 1] = -(positions[:, 2] - center[2])
        rx[:, 2] = positions[:, 1] - center[1]
        ry[:, 0] = positions[:, 2] - center[2]
        ry[:, 2] = -(positions[:, 0] - center[0])
        rz[:, 0] = -(positions[:, 1] - center[1])
        rz[:, 1] = positions[:, 0] - center[0]

        # Normalizing transl., rotat. constraints
        for r in list2_constraints:
            r /= torch.linalg.norm(r.ravel())

        # Add all angle, etc. constraint vectors
        for constraint in self.constraints:
            constraint.setup_jacobian(positions)
            constraint.adjust_forces(positions, forces)
            list_constraints.insert(0, constraint.jacobian)
        # QR DECOMPOSITION - GRAM SCHMIDT

        list_constraints = [r.ravel() for r in list_constraints]
        aa = torch.column_stack(list_constraints)
        (aa, bb) = torch.linalg.qr(aa)
        # Projection
        hh = []
        for i, constraint in enumerate(self.constraints):
            hh.append(aa[:, i] * aa[:, i].reshape(-1, 1))

        txx = aa[:, self.n] * (aa[:, self.n]).reshape(-1, 1)
        tyy = aa[:, self.n + 1] * (aa[:, self.n + 1]).reshape(-1, 1)
        tzz = aa[:, self.n + 2] * (aa[:, self.n + 2]).reshape(-1, 1)
        rxx = aa[:, self.n + 3] * (aa[:, self.n + 3]).reshape(-1, 1)
        ryy = aa[:, self.n + 4] * (aa[:, self.n + 4]).reshape(-1, 1)
        rzz = aa[:, self.n + 5] * (aa[:, self.n + 5]).reshape(-1, 1)
        T = txx + tyy + tzz + rxx + ryy + rzz
        for vec in hh:
            T += vec

        ff = torch.matmul(T, ff).reshape(-1, 1)
        forces[:, :] -= torch.matmul(T, ff).reshape(-1, 1).reshape(-1, 3)

    def __repr__(self):
        constraints = repr(self.constraints)
        return 'FixInternals(_copy_init=%s, epsilon=%s)' % (constraints,
                                                            repr(self.epsilon))

    def __str__(self):
        return '\n'.join([repr(c) for c in self.constraints])

    # Classes for internal use in FixInternals
    class FixInternalsBase:
        """Base class for subclasses of FixInternals."""

        def __init__(self,
                     device,
                     targetvalue,
                     indices,
                     masses,
                     cell,
                     pbc):

            self.device = device
            self.targetvalue = targetvalue  # constant target value

            # indices, defs

            self.indices = torch.stack([defin[0:-1] for defin in indices])
            self.coefs = torch.Tensor([defin[-1]
                                       for defin in indices]).to(self.device)  # coefs
            self.masses = masses
            self.jacobian = []  # geometric Jacobian matrix, Wilson B-matrix
            self.sigma = 1.  # difference between current and target value
            self.projected_force = None  # helps optimizers scan along constr.
            self.cell = cell
            self.pbc = pbc

        def finalize_jacobian(self, pos, n_internals, n, derivs):
            """Populate jacobian with derivatives for `n_internals` defined
            internals. n = 2 (bonds), 3 (angles), 4 (dihedrals)."""
            jacobian = torch.zeros((n_internals, *pos.shape)).to(self.device)
            for i, idx in enumerate(self.indices):
                for j in range(n):
                    jacobian[i, idx[j]] = derivs[i, j]
            jacobian = jacobian.reshape((n_internals, 3 * len(pos)))
            return self.coefs @ jacobian

        def finalize_positions(self, newpos):

            jacobian = self.jacobian / self.masses
            lamda = -self.sigma / (jacobian @ self.get_jacobian(newpos))
            dnewpos = lamda * jacobian

            newpos += dnewpos.reshape(newpos.shape)

        def adjust_forces(self, positions, forces):
            self.projected_force = torch.dot(self.jacobian, forces.ravel())
            self.jacobian /= torch.linalg.norm(self.jacobian)

    class FixBondCombo(FixInternalsBase):
        """Constraint subobject for fixing linear combination of bond lengths
        within FixInternals.

        sum_i( coef_i * bond_length_i ) = constant
        """

        def get_jacobian(self, pos):

            bondvectors = pos[self.indices[:, 1]] - pos[self.indices[:, 0]]
            derivs = get_distances_derivatives(bondvectors, cell=self.cell,
                                               pbc=self.pbc)
            return self.finalize_jacobian(pos, len(bondvectors), 2, derivs)

        def setup_jacobian(self, pos):
            self.jacobian = self.get_jacobian(pos)

        def adjust_positions(self, oldpos, newpos):
            bondvectors = (newpos[self.indices[:, 1]] -
                           newpos[self.indices[:, 0]])

            if self.cell is None:
                dists = torch.linalg.norm(bondvectors, axis=-1)
            else:
                raise NotImplementedError("Not yet implemented for PBC")

            value = self.coefs @ dists
            self.sigma = value - self.targetvalue

            self.finalize_positions(newpos)

        def __repr__(self):
            return 'FixBondCombo({}, {}, {})'.format(repr(self.targetvalue),
                                                     self.indices, self.coefs)

    class FixBondLengthAlt(FixBondCombo):
        """Constraint subobject for fixing bond length within FixInternals.
        Fix distance between atoms with indices a1, a2."""

        def __init__(self, device, targetvalue, indices, masses, cell, pbc):
            if targetvalue <= 0.:
                raise ZeroDivisionError('Invalid targetvalue for fixed bond')

            # bond definition with coef 1.
            indices = torch.LongTensor([list(indices) + [1.]])
            super().__init__(device, targetvalue, indices, masses, cell=cell, pbc=pbc)

        def __repr__(self):
            return 'FixBondLengthAlt({}, {})'.format(self.targetvalue,
                                                     *self.indices)

    class FixAngle(FixInternalsBase):
        """Constraint subobject for fixing an angle within FixInternals.

        Convergence is potentially problematic for angles very close to
        0 or 180 degrees as there is a singularity in the Cartesian derivative.
        Fixing planar angles is therefore not supported at the moment.
        """

        def __init__(self, targetvalue, indices, masses, cell, pbc):
            """Fix atom movement to construct a constant angle."""
            if targetvalue <= 0. or targetvalue >= 180.:
                raise ZeroDivisionError('Invalid targetvalue for fixed angle')

            # angle definition with coef 1.
            indices = torch.LongTensor([list(indices) + [1.]])
            super().__init__(targetvalue, indices, masses, cell=cell, pbc=pbc)

        def gather_vectors(self, pos):
            v0 = [pos[h] - pos[k] for h, k, l in self.indices]
            v1 = [pos[l] - pos[k] for h, k, l in self.indices]
            return v0, v1

        def get_jacobian(self, pos):
            v0, v1 = self.gather_vectors(pos)
            derivs = get_angles_derivatives(v0, v1, cell=self.cell,
                                            pbc=self.pbc)
            return self.finalize_jacobian(pos, len(v0), 3, derivs)

        def setup_jacobian(self, pos):
            self.jacobian = self.get_jacobian(pos)

        def adjust_positions(self, oldpos, newpos):
            v0, v1 = self.gather_vectors(newpos)
            value = get_angles(v0, v1, cell=self.cell, pbc=self.pbc)
            self.sigma = value - self.targetvalue
            self.finalize_positions(newpos)

        def __repr__(self):
            return 'FixAngle({}, {})'.format(self.targetvalue, *self.indices)

    class FixDihedral(FixInternalsBase):
        """Constraint subobject for fixing a dihedral angle within FixInternals.

        A dihedral becomes undefined when at least one of the inner two angles
        becomes planar. Make sure to avoid this situation.
        """

        def __init__(self, targetvalue, indices, masses, cell, pbc):
            # dihedral def. with coef 1.
            indices = torch.LongTensor([list(indices) + [1.]])
            super().__init__(targetvalue, indices, masses, cell=cell, pbc=pbc)

        def gather_vectors(self, pos):
            v0 = [pos[k] - pos[h] for h, k, l, m in self.indices]
            v1 = [pos[l] - pos[k] for h, k, l, m in self.indices]
            v2 = [pos[m] - pos[l] for h, k, l, m in self.indices]
            return v0, v1, v2

        def get_jacobian(self, pos):
            v0, v1, v2 = self.gather_vectors(pos)
            derivs = get_dihedrals_derivatives(v0, v1, v2, cell=self.cell,
                                               pbc=self.pbc)
            return self.finalize_jacobian(pos, len(v0), 4, derivs)

        def setup_jacobian(self, pos):
            self.jacobian = self.get_jacobian(pos)

        def adjust_positions(self, oldpos, newpos):
            v0, v1, v2 = self.gather_vectors(newpos)
            value = get_dihedrals(v0, v1, v2, cell=self.cell, pbc=self.pbc)
            # apply minimum dihedral difference 'convention': (diff <= 180)
            self.sigma = (value - self.targetvalue + 180) % 360 - 180
            self.finalize_positions(newpos)

        def __repr__(self):
            return 'FixDihedral({}, {})'.format(self.targetvalue, *self.indices)
