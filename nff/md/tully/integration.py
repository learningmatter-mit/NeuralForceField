import numpy as np
import pickle

from nff.utils import constants as const
from nff.md.utils_ax import atoms_to_nxyz
from nff.md.tully.io import get_results
from nff.md.tully.step import (runge_c, try_hop,
                               verlet_step_1, verlet_step_2)


class NeuralTully:
    def __init__(self,
                 atoms_list,
                 cutoff,
                 cutoff_skin,
                 device,
                 batch_size,
                 num_states,
                 surf,
                 nuc_dt,
                 elec_dt,
                 max_gap_hop,
                 max_time,
                 nbr_update_period,
                 save_file,
                 minimal_save_file,
                 log_file,
                 save_period):
        """
        `max_gap_hop` in a.u.
        """

        self.atoms_list = atoms_list
        self.vel = self.get_vel()
        # terms for decoherence correction
        self.delta_R = 0
        self.delta_P = 0
        self.c = self.init_c()
        self.T = None

        self.t = 0
        self.props = None
        self.num_atoms = [len(atoms) for atoms
                          in self.atoms_list]
        self.num_samples = len(atoms_list)

        self.cutoff = cutoff
        self.cutoff_skin = cutoff_skin
        self.device = device
        self.batch_size = batch_size
        self.num_states = num_states
        self.surfs = np.ones_like(self.num_samples) * surf
        self.elec_dt = elec_dt * const.FS_TO_AU
        self.elec_nuc_scale = (nuc_dt // elec_dt)
        self.nuc_dt = self.elec_nuc_scale * elec_dt * const.FS_TO_AU

        self.max_time = max_time * const.FS_TO_AU
        self.nbr_update_period = nbr_update_period
        self.nbr_list = None
        self.max_gap_hop = max_gap_hop

        self.save_file = save_file
        self.minimal_save_file = minimal_save_file
        self.log_file = log_file
        self.save_period = save_period

        self.log_template = self.setup_logging()

    def get_vel(self):
        vel = np.stack([atoms.get_velocities()
                        for atoms in self.atoms_list])
        vel /= const.BOHR_RADIUS * const.ASE_TO_FS * const.FS_TO_AU

        return vel

    def init_c(self):
        c = np.zeros((self.num_samples, self.num_states))
        c[:, self.surfs[0]] = 1
        return c

    @property
    def U(self):
        if not self.props:
            return
        return self.props["U"]

    @property
    def forces(self):
        _forces = np.stack([-self.props[f'energy_{i}_grad']
                            for i in range(self.num_states)],
                           axis=1)
        return _forces

    @property
    def energy(self):
        _energy = np.stack([-self.props[f'energy_{i}']
                            for i in range(self.num_states)],
                           axis=1)
        return _energy

    @property
    def nacv(self):
        _nacv = np.zeros((self.num_samples, self.num_states,
                          self.num_states, self.num_atoms, 3))
        for i in range(self.num_states):
            for j in range(self.num_states):
                if i == j:
                    continue
                _nacv[:, i, j, :] = self.props[f'nacv_{i}{j}']
        return _nacv

    @property
    def mass(self):
        _mass = (self.atoms_list[0].get_masses()
                 * const.AMU_TO_AU)
        return _mass

    @property
    def nxyz(self):
        _nxyz = np.stack([atoms_to_nxyz(atoms) for atoms in
                          self.atoms_list])

        return _nxyz

    @property
    def xyz(self):
        _xyz = self.nxyz[..., 1:]
        return _xyz

    @xyz.setter
    def xyz(self, val):
        for atoms, xyz in zip(self.atoms_list, val):
            atoms.set_positions(xyz)

    @property
    def state_dict(self):
        _state_dict = {"nxyz": self.nxyz,
                       "nacv": self.nacv,
                       "energy": self.energy,
                       "forces": self.forces,
                       "U": self.U,
                       "t": self.t,
                       "vel": self.vel,
                       "c": self.c,
                       "T": self.T,
                       "surfs": self.surfs}
        return _state_dict

    @property
    def minimal_state_dict(self):
        keys = ['nxyz', 'energy', 'surfs', 't']
        minimal = {key: self.state_dict[key]
                   for key in keys}

        return minimal

    def save(self):
        with open(self.save_file, "ab") as f:
            pickle.dump(self.state_dict, f)

        with open(self.minimal_save_file, "ab") as f:
            pickle.dump(self.minimal_state_dict, f)

    def setup_logging(self):

        states = [f"State {i}" for i in range(self.num_states)]
        hdr = "%-9s " % "Time [ps]"
        for state in states:
            hdr += "%15s " % state

        with open(self.log_file, 'w') as f:
            f.write(hdr)

        template = "%-10.4f "
        for state in states:
            template += "%12.4f%%"

        return template

    def log(self):
        time = self.t / const.FS_TO_AU / 1000
        pcts = []
        for i in range(self.num_states):
            num_surf = (self.surfs == i).sum()
            pct = num_surf / self.num_samples * 100
            pcts.append(pct)

        text = self.template % (time, *pcts)

        with open(self.log_file, 'a') as f:
            f.write(text)

    @classmethod
    def from_file(cls,
                  file,
                  max_time=None):

        state_dicts = []
        with open(file, 'rb') as f:
            while True:
                try:
                    state_dict = pickle.load(f)
                    state_dicts.append(state_dict)
                except EOFError:
                    break
                time = state_dict['t']
                if max_time is not None and time > max_time:
                    break
        return state_dicts

    def update_props(self,
                     needs_nbrs):

        props = get_results(model=self.model,
                            nxyz=self.nxyz,
                            nbr_list=self.nbr_list,
                            num_atoms=self.num_atoms,
                            needs_nbrs=needs_nbrs,
                            cutoff=self.cutoff,
                            cutoff_skin=self.cutoff_skin,
                            device=self.device,
                            batch_size=self.batch_size,
                            old_U=self.U,
                            num_states=self.num_states,
                            surf=self.surfs,
                            max_gap_hop=self.max_gap_hop)

        self.props = props

    def step(self, needs_nbrs):

        # inner loop to propagate electronic wavefunction
        for _ in range(self.elec_nuc_scale):
            c, T = runge_c(c=self.c,
                           vel=self.vel,
                           results=self.props,
                           elec_dt=self.elec_dt)

            new_surfs, new_vel = try_hop(c=c,
                                         T=T,
                                         dt=self.elec_dt,
                                         surfs=self.surfs,
                                         vel=self.vel,
                                         nacv=self.nacv,
                                         mass=self.mass,
                                         energy=self.energy)

            # if any sample hopped then it was within the cutoff
            # gap, which means we have forces for the other states
            # which we can use for the rest of this inner loop

            self.surfs = new_surfs
            self.vel = new_vel
            self.c = c
            self.T = T
            self.t += self.elec_dt

        # outer loop to move nuclei
        # xyz converted to a.u. for the step and then
        # back to Angstrom after
        new_xyz, new_vel = verlet_step_1(self.forces,
                                         self.surfs,
                                         vel=self.vel,
                                         xyz=self.xyz / const.BOHR_RADIUS,
                                         mass=self.mass,
                                         nuc_dt=self.nuc_dt)
        self.xyz = new_xyz * const.BOHR_RADIUS
        self.vel = new_vel

        self.update_props(needs_nbrs)
        new_vel = verlet_step_2(forces=self.forces,
                                surfs=self.surfs,
                                vel=self.vel,
                                xyz=self.xyz,
                                mass=self.mass,
                                nuc_dt=self.nuc_dt)
        self.vel = new_vel
        self.log()

    def run(self):
        steps = self.max_time // self.nuc_dt
        epochs = steps // self.nbr_update_period

        counter = 0
        for _ in range(epochs):
            for i in range(self.nbr_update_period):
                needs_nbrs = (i == 0)
                self.step(needs_nbrs=needs_nbrs)

                counter += 1
                if counter % self.save_period == 0:
                    self.save()
