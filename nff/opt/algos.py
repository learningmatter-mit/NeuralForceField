from ase.optimize.sciopt import SciPyFminCG, SciPyFminBFGS
from ase.optimize import BFGS
import scipy.optimize as opt
import numpy as np


class Converged(Exception):
    pass


class OptimizerConvergenceError(Exception):
    pass


class NeuralCG(SciPyFminCG):
    def call_fmin(self, fmax, steps):
        output = opt.fmin_cg(
            self.f,
            self.x0(),
            fprime=self.fprime,
            # args=(),
            gtol=fmax * 0.1,  # Should never be reached
            norm=np.inf,
            # epsilon=
            maxiter=steps,
            full_output=1,
            disp=0,
            # retall=0,
            callback=self.callback,
        )
        warnflag = output[-1]
        if warnflag == 2:
            raise OptimizerConvergenceError("Warning: Desired error not necessarily achieved " "due to precision loss")

    def run(self, fmax=0.05, steps=100000000):
        if self.force_consistent is None:
            self.set_force_consistent()
        self.fmax = fmax
        try:
            # want to update the neighbor list every step
            self.atoms.update_nbr_list()

            # As SciPy does not log the zeroth iteration, we do that manually
            self.callback(None)

            # Scale the problem as SciPy uses I as initial Hessian.
            self.call_fmin(fmax / self.H0, steps)
        except Converged:
            pass


class NeuralBFGS(SciPyFminBFGS):
    def run(self, fmax=0.05, steps=100000000):
        if self.force_consistent is None:
            self.set_force_consistent()
        self.fmax = fmax
        try:
            # want to update the neighbor list every step
            self.atoms.update_nbr_list()

            # As SciPy does not log the zeroth iteration, we do that manually
            self.callback(None)
            # Scale the problem as SciPy uses I as initial Hessian.
            self.call_fmin(fmax / self.H0, steps)
        except Converged:
            pass


class NeuralAseBFGS(BFGS):
    def step(self, f=None):
        atoms = self.atoms

        atoms.update_nbr_list()

        if f is None:
            f = atoms.get_forces()

        r = atoms.get_positions()
        f = f.reshape(-1)
        self.update(r.flat, f, self.r0, self.f0)

        from numpy.linalg import eigh

        omega, V = eigh(self.H)

        # FUTURE: Log this properly
        # # check for negative eigenvalues of the hessian
        # if any(omega < 0):
        #     n_negative = len(omega[omega < 0])
        #     msg = '\n** BFGS Hessian has {} negative eigenvalues.'.format(
        #         n_negative
        #     )
        #     print(msg, flush=True)
        #     if self.logfile is not None:
        #         self.logfile.write(msg)
        #         self.logfile.flush()

        dr = np.dot(V, np.dot(f, V) / np.fabs(omega)).reshape((-1, 3))
        steplengths = (dr**2).sum(1) ** 0.5
        dr = self.determine_step(dr, steplengths)
        atoms.set_positions(r + dr)
        self.r0 = r.flat.copy()
        self.f0 = f.copy()
        self.dump((self.H, self.r0, self.f0, self.maxstep))
