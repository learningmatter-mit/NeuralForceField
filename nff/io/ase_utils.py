import numpy as np
import torch
import time

from ase.constraints import FixConstraint
from ase.geometry import get_dihedrals_derivatives, get_angles_derivatives
from ase.optimize import BFGS, LBFGS

from nff.utils.scatter import scatter_add


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


def get_bond_forces(atoms,
                    idx,
                    k,
                    length_0):

    deltas = (atoms.get_positions()[idx[:, 0]] -
              atoms.get_positions()[idx[:, 1]])
    bond_lens = np.linalg.norm(deltas, axis=-1)

    forces_0 = -2 * k.reshape(-1, 1) * deltas * ((-length_0 + bond_lens) /
                                                 bond_lens).reshape(-1, 1)
    forces_1 = -forces_0

    total_forces = np.zeros_like(atoms.get_positions())
    total_forces[idx[:, 0]] += forces_0
    total_forces[idx[:, 1]] += forces_1

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

    def __str__(self):
        return repr(self)

    def __repr__(self):
        idx_str = str(self.idx)
        val_str = str(np.degrees(self.targ_angles))

        return 'Constrain angles (indices=%s, values (deg.)=%s)' % (idx_str, val_str)


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

    def __str__(self):
        return repr(self)

    def __repr__(self):
        idx_str = str(self.idx)
        val_str = str(np.degrees(self.targ_diheds))

        return 'Constrain dihedrals (indices=%s, values (deg.)=%s)' % (idx_str, val_str)


class ConstrainBonds(FixConstraint):

    def __init__(self,
                 idx,
                 atoms,
                 force_consts,
                 targ_lengths=None):

        self.idx = np.asarray(idx)

        if targ_lengths is not None:
            self.targ_lengths = np.array(targ_lengths).reshape(-1)
        else:
            deltas = (atoms.get_positions()[idx[:, 0]] -
                      atoms.get_positions()[idx[:, 1]])
            self.targ_lengths = np.linalg.norm(deltas, axis=-1)

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
        new_forces = get_bond_forces(atoms=atoms,
                                     idx=self.idx,
                                     k=self.force_consts,
                                     length_0=self.targ_lengths)

        forces += new_forces

        return new_forces, forces

    def __str__(self):
        return repr(self)

    def __repr__(self):
        idx_str = str(self.idx)
        val_str = str(self.targ_lengths)

        return 'Constrain bonds (indices=%s, values=%s)' % (idx_str, val_str)


def split(array, num_atoms):
    shape = [-1]
    total_atoms = num_atoms.sum()
    if not all([i == total_atoms for i in np.array(array).shape]):
        shape = [-1, 3]

    split_idx = np.cumsum(num_atoms)
    split_array = np.split(np.array(array).reshape(*shape), split_idx)[:-1]

    return split_array


class BatchedBFGS(BFGS):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_atoms = self.atoms.num_atoms.numpy().astype('int')
        self.save = kwargs.get("save", False)

        self.split_h0()

    def determine_step(self, dr, steplengths, f):

        # scale steps of different batches separately

        steplength_by_batch = split(array=steplengths,
                                    num_atoms=self.num_atoms)
        dr_by_batch = split(dr, self.num_atoms)
        maxsteplengths = [np.max(i) for i in steplength_by_batch]

        for i, this_max in enumerate(maxsteplengths):
            if this_max >= self.maxstep:
                scale = self.maxstep / this_max
                dr_by_batch[i] *= scale

        # don't update any batch that has f <= self.fmax

        f_by_batch = split(array=f,
                           num_atoms=self.num_atoms)
        for i, this_f in enumerate(f_by_batch):
            this_f_max = ((this_f ** 2).sum(axis=1) ** 0.5).max()
            if this_f_max < self.fmax:
                dr_by_batch[i] *= 0

        dr = np.concatenate(dr_by_batch)

        return dr

    def step(self, f=None):
        atoms = self.atoms

        if f is None:
            f = atoms.get_forces()

        r = atoms.get_positions()
        f = f.reshape(-1)
        self.update(r.flat, f, self.r0, self.f0)

        drs = []
        start = 0

        for i, H in enumerate(self.H):

            num_atoms = self.num_atoms[i]
            delta = num_atoms * 3
            stop = start + delta

            omega, V = np.linalg.eigh(H)
            this_dr = np.dot(V, np.dot(f[start: stop], V) /
                             np.fabs(omega)).reshape((-1, 3))
            drs.append(this_dr)

            start += delta

        dr = np.concatenate(drs)
        steplengths = (dr**2).sum(1)**0.5
        dr = self.determine_step(dr, steplengths, f)
        atoms.set_positions(r + dr)
        self.r0 = r.flat.copy()
        self.f0 = f.copy()

        # takes up a lot of unnecessary time when doing batched opt
        if self.save:
            self.dump((self.H, self.r0, self.f0, self.maxstep))

    def split_h0(self):

        new_h = []
        counter = 0
        for num in self.num_atoms:
            start = counter
            delta = 3 * num
            stop = start + delta

            new_h.append(self.H0[start: stop, start: stop])

            counter += delta

        self.H0 = new_h

    def update(self, r, f, r0, f0):

        # copied from original, but with modification for test of np.abs(dr).max()
        if self.H is None:
            self.H = self.H0
            return

        split_f = split(f, self.num_atoms)
        split_f0 = split(f0, self.num_atoms)
        split_r = split(r, self.num_atoms)
        split_r0 = split(r0, self.num_atoms)

        for i, this_r in enumerate(split_r):

            this_r0 = split_r0[i]
            this_f = split_f[i]
            this_f0 = split_f0[i]
            this_dr = (this_r - this_r0).reshape(-1)

            if np.abs(this_dr).max() < 1e-7:
                continue

            df = (this_f - this_f0).reshape(-1)
            a = np.dot(this_dr, df)
            dg = np.dot(self.H[i], this_dr)
            b = np.dot(this_dr, dg)

            self.H[i] -= (np.outer(df, df) /
                          a + np.outer(dg, dg) / b)

    def converged(self,
                  forces=None):

        if forces is None:
            forces = self.atoms.get_forces()

        # set nans to zero because they'll never converge anyway, so might as well
        # stop the opt when everything else is converged

        forces[np.bitwise_not(np.isfinite(forces))] = 0

        if hasattr(self.atoms, "get_curvature"):
            return (forces ** 2).sum(
                axis=1
            ).max() < self.fmax ** 2 and self.atoms.get_curvature() < 0.0
        return (forces ** 2).sum(axis=1).max() < self.fmax ** 2

    def log(self, forces=None):
        if forces is None:
            forces = self.atoms.get_forces()

        fmax = [np.sqrt((i ** 2).sum(axis=1).max())
                for i in split(forces, self.num_atoms)]

        e = self.atoms.get_potential_energy(
            force_consistent=self.force_consistent
        )

        t = time.localtime()

        if self.logfile is not None:
            name = self.__class__.__name__
            num_batches = len(self.num_atoms)

            if self.nsteps == 0:

                args = [" " * len(name), "Step", "Time"]
                args += ["Energy %d" % (i + 1) for i in range(num_batches)]
                args += ["fmax %d" % (i + 1) for i in range(num_batches)]
                args = tuple(args)

                msg = "%s  %4s %8s "
                msg += "%15s " * num_batches
                msg += "%14s  " * num_batches
                msg += "\n"

                msg = msg % args
                self.logfile.write(msg)

            ast = ''
            args = [name, self.nsteps, t[3], t[4], t[5]]
            args += e.tolist()
            args += [ast]
            args += fmax

            args = tuple(args)

            msg = "%s:  %3d %02d:%02d:%02d "
            msg += "%15.6f " * num_batches
            msg = msg[:-1] + "%1s "
            msg += "%15.6f " * num_batches
            msg += "\n"

            msg = msg % args
            self.logfile.write(msg)
            self.logfile.flush()


class BatchedLBFGS(LBFGS):
    """
    The Hessian is not diagonalized in LBFGS, so each step is faster than in BFGS.
    The diagonalizations happen in serial for batched BFGS, which can make them
    a bottleneck. Avoiding diagonalizations is therefore helpful when doing batched
    optimization.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_atoms = self.atoms.num_atoms.numpy().astype('int')
        self.mol_idx = self.make_mol_idx()
        self.save = kwargs.get("save", False)
        self.memory = kwargs.get("memory", 30)

    def make_mol_idx(self):
        mol_idx = []
        for i, num_atoms in enumerate(self.num_atoms):
            mol_idx += [i] * int(num_atoms) * 3
        mol_idx = torch.LongTensor(mol_idx)

        return mol_idx

    def step(self,
             f=None):
        """
        Take a single step

        Use the given forces, update the history and calculate the next step --
        then take it
        """

        if f is None:
            f = self.atoms.get_forces()

        r = self.atoms.get_positions()

        self.update(r, f, self.r0, self.f0)

        s = np.array(self.s)
        y = np.array(self.y)
        rho = np.array(self.rho)

        h0 = self.H0
        loopmax = np.min([self.memory, self.iteration])
        a = np.empty((loopmax, self.num_atoms.shape[0]), dtype=np.float64)

        # The algorithm itself:

        q = -f.reshape(-1)

        for i in range(loopmax - 1, -1, -1):
            dot = scatter_add(src=torch.Tensor(s[i] * q),
                              index=self.mol_idx,
                              dim=0,
                              dim_size=int(self.mol_idx.max() + 1)).numpy()

            a[i] = rho[i] * dot
            prod = np.repeat(a[i], self.num_atoms * 3) * y[i]
            q -= prod

        z = h0 * q

        for i in range(loopmax):
            dot = scatter_add(src=torch.Tensor(y[i] * z),
                              index=self.mol_idx,
                              dim=0,
                              dim_size=int(self.mol_idx.max() + 1)).numpy()

            b = rho[i] * dot
            delta = a[i] - b
            prod = np.repeat(delta, self.num_atoms * 3) * s[i]
            z += prod

        self.p = - z.reshape((-1, 3))

        g = -f
        if self.use_line_search:
            raise NotImplementedError("Not yet implemented wdith line search")
        else:
            self.force_calls += 1
            self.function_calls += 1
            steplengths = (self.p ** 2).sum(1) ** 0.5
            dr = self.determine_step(dr=self.p,
                                     steplengths=steplengths,
                                     f=f) * self.damping

        self.atoms.set_positions(r + dr)

        self.iteration += 1
        self.r0 = r
        self.f0 = -g

        if self.save:
            self.dump((self.iteration, self.s, self.y,
                       self.rho, self.r0, self.f0, self.e0, self.task))

    def determine_step(self, dr, steplengths, f):
        return BatchedBFGS.determine_step(self=self,
                                          dr=dr,
                                          steplengths=steplengths,
                                          f=f)

    def update(self, r, f, r0, f0):
        """
        Update everything that is kept in memory
        """
        if self.iteration > 0:
            s0 = r.reshape(-1) - r0.reshape(-1)
            self.s.append(s0)

            # We use the gradient which is minus the force!
            y0 = f0.reshape(-1) - f.reshape(-1)
            self.y.append(y0)

            dot = scatter_add(src=torch.Tensor(y0 * s0),
                              index=self.mol_idx,
                              dim=0,
                              dim_size=int(self.mol_idx.max() + 1)).numpy()

            # for anything that's converged and hence not updated
            # (so y0 and s0 are both 0)
            dot[dot == 0] = 1e-13

            rho0 = 1.0 / dot

            self.rho.append(rho0)

        if self.iteration > self.memory:
            self.s.pop(0)
            self.y.pop(0)
            self.rho.pop(0)

    def converged(self,
                  forces=None):
        return BatchedBFGS.converged(self, forces)

    def log(self, forces=None):
        return BatchedBFGS.log(self, forces)
