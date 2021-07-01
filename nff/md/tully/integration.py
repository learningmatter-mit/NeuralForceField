import numpy as np
import random
import copy

from nff.utils import constants as const
from nff.md.utils_ax import atoms_to_nxyz
from nff.md.tully.io import (get_results, get_dc_dt,
                             get_p_hop)
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
                 max_gap_hop):
        """
        `max_gap_hop` in a.u.
        """

        self.atoms_list = atoms_list
        self.num_atoms = [len(atoms) for atoms
                          in self.atoms_list]
        self.num_samples = len(atoms_list)

        self.cutoff = cutoff
        self.cutoff_skin = cutoff_skin
        self.device = device
        self.batch_size = batch_size
        self.num_states = num_states
        self.surfs = np.ones_like(self.num_samples) * surf
        self.elec_dt = elec_dt
        self.elec_nuc_scale = (nuc_dt // elec_dt)
        self.nuc_dt = self.elec_nuc_scale * elec_dt

        self.t = 0
        self.nbr_list = None
        self.props_list = [{"c": self.init_c()}]
        self.max_gap_hop = max_gap_hop

    def init_c(self):
        c = np.zeros((self.num_samples, self.num_states))
        c[:, self.surfs[0]] = 1
        return c

    @property
    def c(self):
        return self.props_list[-1]["c"]

    @c.setter
    def c(self, value):
        self.props_list[-1]["c"] = value

    @property
    def nxyz_list(self):
        lst = [atoms_to_nxyz(atoms) for atoms in
               self.atoms_list]

        return lst

    @property
    def old_U(self):
        if not self.props_list:
            return
        return self.props_list[-1]["U"]

    @property
    def vel(self):
        _vel = np.stack([atoms.get_velocities()
                         for atoms in self.atoms_list])
        return _vel

    @property
    def props(self):
        return self.props_list[-1]

    @props.setter
    def props(self, value):
        self.props_list.append(value)

    @property
    def forces(self):
        _forces = np.stack([-self.props[f'energy_{i}_grad']
                            for i in range(self.num_states)],
                           axis=1)
        return _forces

    @property
    def mass(self):
        _mass = (self.atoms_list[0].get_masses()
                 * const.AMU_TO_AU)
        return _mass

    @property
    def xyz(self):
        _xyz = np.stack(self.nxyz_list)[..., 1:]
        return _xyz

    @xyz.setter
    def xyz(self, val):
        for atoms, xyz in zip(self.atoms_list, val):
            atoms.set_positions(xyz)

    @vel.setter
    def vel(self, val):
        for atoms, this_vel in zip(self.atoms_list, val):
            atoms.set_velocities(this_vel)

    def update_props(self,
                     needs_nbrs):

        props = get_results(model=self.model,
                            nxyz_list=self.nxyz_list,
                            nbr_list=self.nbr_list,
                            num_atoms=self.num_atoms,
                            needs_nbrs=needs_nbrs,
                            cutoff=self.cutoff,
                            cutoff_skin=self.cutoff_skin,
                            device=self.device,
                            batch_size=self.batch_size,
                            old_U=self.old_U,
                            num_states=self.num_states,
                            surf=self.surfs,
                            max_gap_hop=self.max_gap_hop)

        self.props = props

    def step(self, needs_nbrs):
        # outer loop to move nuclei
        new_xyz, new_vel = verlet_step_1(self.forces,
                                         self.surfs,
                                         vel=self.vel,
                                         xyz=self.xyz,
                                         mass=self.mass,
                                         nuc_dt=self.nuc_dt)
        self.xyz = new_xyz
        self.vel = new_vel

        # inner loop to propagate electronic wavefunction
        for _ in range(self.elec_nuc_scale):
            c = runge_c(c=self.c,
                        vel=self.vel,
                        results=self.props,
                        elec_dt=self.elec_dt)
            self.c = c
            self.t += self.elec_dt

        self.update_props(needs_nbrs)
        new_vel = verlet_step_2(forces=self.forces,
                                surfs=self.surfs,
                                vel=self.vel,
                                xyz=self.xyz,
                                mass=self.mass,
                                nuc_dt=self.nuc_dt)
        self.vel = new_vel

    # def step(self,
    #          needs_nbrs):

    #     self.update_props(needs_nbrs)

    #     dc_dt, T = get_dc_dt(c=self.c,
    #                          vel=self.vel,
    #                          results=self.props,
    #                          num_states=self.num_states)

    #     # need to step c, the positions, and the velocities,
    #     # and use different time-steps for each

    #     c = self.step_c(dc_dt)

    #     p_hop = get_p_hop(c=c,
    #                       T=T,
    #                       dt=self.dt,
    #                       surf=self.surf)

    #     new_surf = self.get_new_surf(p_hop)
    #     if new_surf != self.surf:
    #         self.rescale()
    #         self.surf = new_surf

    #     self.decoherence()
