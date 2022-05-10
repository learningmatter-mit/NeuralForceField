import numpy as np
import torch
import time

from ase.constraints import FixConstraint
from ase.geometry import get_dihedrals_derivatives, get_angles_derivatives
from ase.optimize import BFGS, LBFGS


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

    def initialize(self):
        """Initialize everything so no checks have to be done in step"""
        self.iteration = 0
        self.s = []
        self.y = []
        # Store also rho, to avoid calculating the dot product again and
        # again.
        self.rho = []

        self.r0 = None
        self.f0 = None
        self.e0 = None
        self.task = 'START'
        self.load_restart = False

    def read(self):
        """Load saved arrays to reconstruct the Hessian"""
        self.iteration, self.s, self.y, self.rho, \
            self.r0, self.f0, self.e0, self.task = self.load()
        self.load_restart = True

    def step(self, f=None):
        """Take a single step

        Use the given forces, update the history and calculate the next step --
        then take it"""

        if f is None:
            f = self.atoms.get_forces()

        r = self.atoms.get_positions()

        self.update(r, f, self.r0, self.f0)

        s = self.s
        y = self.y
        rho = self.rho
        h0 = self.H0

        loopmax = np.min([self.memory, self.iteration])
        a = np.empty((loopmax,), dtype=np.float64)

        # ## The algorithm itself:
        q = -f.reshape(-1)
        for i in range(loopmax - 1, -1, -1):
            a[i] = rho[i] * np.dot(s[i], q)
            q -= a[i] * y[i]
        z = h0 * q

        for i in range(loopmax):
            b = rho[i] * np.dot(y[i], z)
            z += s[i] * (a[i] - b)

        self.p = - z.reshape((-1, 3))
        # ##

        g = -f
        if self.use_line_search is True:
            e = self.func(r)
            self.line_search(r, g, e)
            dr = (self.alpha_k * self.p).reshape(len(self.atoms), -1)
        else:
            self.force_calls += 1
            self.function_calls += 1
            dr = self.determine_step(self.p) * self.damping
        self.atoms.set_positions(r + dr)

        self.iteration += 1
        self.r0 = r
        self.f0 = -g

        if self.save:
            self.dump((self.iteration, self.s, self.y,
                       self.rho, self.r0, self.f0, self.e0, self.task))

    def determine_step(self, dr):
        """Determine step to take according to maxstep

        Normalize all steps as the largest step. This way
        we still move along the eigendirection.
        """
        steplengths = (dr**2).sum(1)**0.5
        longest_step = np.max(steplengths)
        if longest_step >= self.maxstep:
            dr *= self.maxstep / longest_step

        return dr

    def update(self, r, f, r0, f0):
        """Update everything that is kept in memory

        This function is mostly here to allow for replay_trajectory.
        """
        if self.iteration > 0:
            s0 = r.reshape(-1) - r0.reshape(-1)
            self.s.append(s0)

            # We use the gradient which is minus the force!
            y0 = f0.reshape(-1) - f.reshape(-1)
            self.y.append(y0)

            rho0 = 1.0 / np.dot(y0, s0)
            self.rho.append(rho0)

        if self.iteration > self.memory:
            self.s.pop(0)
            self.y.pop(0)
            self.rho.pop(0)

    def replay_trajectory(self, traj):
        """Initialize history from old trajectory."""
        if isinstance(traj, str):
            from ase.io.trajectory import Trajectory
            traj = Trajectory(traj, 'r')
        r0 = None
        f0 = None
        # The last element is not added, as we get that for free when taking
        # the first qn-step after the replay
        for i in range(0, len(traj) - 1):
            r = traj[i].get_positions()
            f = traj[i].get_forces()
            self.update(r, f, r0, f0)
            r0 = r.copy()
            f0 = f.copy()
            self.iteration += 1
        self.r0 = r0
        self.f0 = f0

    def func(self, x):
        """Objective function for use of the optimizers"""
        self.atoms.set_positions(x.reshape(-1, 3))
        self.function_calls += 1
        return self.atoms.get_potential_energy(
            force_consistent=self.force_consistent)

    def fprime(self, x):
        """Gradient of the objective function for use of the optimizers"""
        self.atoms.set_positions(x.reshape(-1, 3))
        self.force_calls += 1
        # Remember that forces are minus the gradient!
        return - self.atoms.get_forces().reshape(-1)

    def line_search(self, r, g, e):
        self.p = self.p.ravel()
        p_size = np.sqrt((self.p**2).sum())
        if p_size <= np.sqrt(len(self.atoms) * 1e-10):
            self.p /= (p_size / np.sqrt(len(self.atoms) * 1e-10))
        g = g.ravel()
        r = r.ravel()
        ls = LineSearch()
        self.alpha_k, e, self.e0, self.no_update = \
            ls._line_search(self.func, self.fprime, r, self.p, g, e, self.e0,
                            maxstep=self.maxstep, c1=.23,
                            c2=.46, stpmax=50.)
        if self.alpha_k is None:
            raise RuntimeError('LineSearch failed!')
