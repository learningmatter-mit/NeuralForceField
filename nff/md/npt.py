import os
import numpy as np
import copy
import math
from tqdm import tqdm
from ase.md.md import MolecularDynamics
from ase.optimize.optimize import Dynamics
from ase.md.npt import NPT
from ase.md.nptberendsen import Inhomogeneous_NPTBerendsen
from ase import units
from ase.md.velocitydistribution import (MaxwellBoltzmannDistribution,
                                         Stationary, ZeroRotation)



class NoseHoovernpt(NPT):
    def __init__(self, atoms,
                 timestep,
                 temperature=None,
                 externalstress=None,
                 ttime=None,
                 T_init=None,
                 pfactor=None,
                 mask=None,
                 trajectory=None,
                 logfile=None,
                 loginterval=1,
                 nbr_update_period=20,
                 append_trajectory=False,
                 **kwargs):

        if os.path.isfile(str(trajectory)):
            os.remove(trajectory)

        print(externalstress)

        NPT.__init__(self,atoms=atoms,
                     timestep=timestep * units.fs,
                     ttime=ttime,
                     externalstress=externalstress,
                     pfactor=pfactor,
                     temperature_K=temperature,
                     mask=mask,
                     trajectory=trajectory,
                     logfile=logfile,
                     loginterval=loginterval,
                     append_trajectory=append_trajectory)

        # Initialize simulation parameters
        # convert units
        self.nbr_update_period = nbr_update_period
        self.max_steps=0

        self.T = temperature * units.kB

        # initial Maxwell-Boltmann temperature for atoms
        if T_init is not None:
            # convert units
            T_init = T_init * units.kB
        else:
            T_init = 2 * self.T

        MaxwellBoltzmannDistribution(self.atoms, T_init)
        Stationary(self.atoms)
        ZeroRotation(self.atoms)
        self.initialize()


    def run(self, steps=None):

        if steps is None:
            steps = self.num_steps

        epochs = math.ceil(steps / self.nbr_update_period)
        # number of steps in between nbr updates
        steps_per_epoch = int(steps / epochs)
        # maximum number of steps starts at `steps_per_epoch`
        # and increments after every nbr list update
        #self.max_steps = 0
        self.atoms.update_nbr_list()

        for _ in tqdm(range(epochs)):
            self.max_steps += steps_per_epoch
            Dynamics.run(self)
            self.atoms.update_nbr_list()



class Berendsennpt(Inhomogeneous_NPTBerendsen):
    def __init__(self, atoms,
                 timestep,
                 temperature=None,
                 taut=0.5e3*units.fs,
                 taup=1e3*units.fs,
                 pressure=1,
                 compressibility=None,
                 T_init=None,
                 mask=(1,1,1),
                 fixcm=True,
                 trajectory=None,
                 logfile=None,
                 loginterval=1,
                 nbr_update_period=20,
                 append_trajectory=False,
                 **kwargs):

        if os.path.isfile(str(trajectory)):
            os.remove(trajectory)

        Inhomogeneous_NPTBerendsen.__init__(self, atoms=atoms,
                                            timestep=timestep * units.fs,
                                            temperature=temperature,
                                            taut=taut,
                                            taup=taup,
                                            pressure=pressure,
                                            compressibility=compressibility,
                                            fixcm=fixcm,
                                            mask=mask,
                                            trajectory=trajectory,
                                            logfile=logfile,
                                            loginterval=loginterval)        

        # Initialize simulation parameters
        # convert units
        self.nbr_update_period = nbr_update_period
        self.max_steps=0

        self.T = temperature * units.kB

        # initial Maxwell-Boltmann temperature for atoms
        if T_init is not None:
            # convert units
            T_init = T_init * units.kB
        else:
            T_init = 2 * self.T

        MaxwellBoltzmannDistribution(self.atoms, T_init)
        Stationary(self.atoms)
        ZeroRotation(self.atoms)

        print(self.atoms.get_temperature())


    def run(self, steps=None):

        if steps is None:
            steps = self.num_steps

        epochs = math.ceil(steps / self.nbr_update_period)
        # number of steps in between nbr updates
        steps_per_epoch = int(steps / epochs)
        # maximum number of steps starts at `steps_per_epoch`
        # and increments after every nbr list update
        #self.max_steps = 0
        self.atoms.update_nbr_list()

        for _ in tqdm(range(epochs)):
            self.max_steps += steps_per_epoch
            Dynamics.run(self)
            self.atoms.update_nbr_list()


            
class NoseHooverNPT(MolecularDynamics):
    def __init__(self,
                 atoms,
                 timestep,
                 temperature,
                 pressure,
                 ttime,
                 Pdamp,
                 maxwell_temp=None,
                 trajectory=None,
                 logfile=None,
                 loginterval=1,
                 max_steps=None,
                 nbr_update_period=20,
                 append_trajectory=True,
                 **kwargs):

        if os.path.isfile(str(trajectory)):
            os.remove(trajectory)

        MolecularDynamics.__init__(self,
                                   atoms=atoms,
                                   timestep=timestep * units.fs,
                                   trajectory=trajectory,
                                   logfile=logfile,
                                   loginterval=loginterval,
                                   append_trajectory=append_trajectory)

        # Initialize simulation parameters
        # convert units

        self.dt = timestep * units.fs
        self.T = temperature * units.kB
        self.P= pressure*units.GPa
        self.ttime = ttime  # defined as a fraction of self.dt
        self.pdamp= Pdamp
        # Q is chosen to be 6 N kT
        self.Natom = len(atoms)
        self.Nf =3*self.Natom - 6

        # no rotation or translation, so target kinetic energy
        # is 1/2 (3N - 6) kT
        self.targeEkin = 0.5 * (self.Nf) * self.T

        self.Q = (self.Nf ) * self.T * (self.ttime * self.dt)**2
        self.W = (self.Natom-1)* self.T *(self.pdamp*self.dt)**2
        self.zeta = 0.0
        self.eta=0.0
        self.veta=0.0
        self.num_steps = max_steps
        self.n_steps = 0
        self.max_steps = 0

        self.nbr_update_period = nbr_update_period

        # initial Maxwell-Boltmann temperature for atoms
        if maxwell_temp is not None:
            # convert units
            maxwell_temp = maxwell_temp * units.kB
        else:
            maxwell_temp = 2 * self.T

        MaxwellBoltzmannDistribution(self.atoms, maxwell_temp)
        Stationary(self.atoms)
        ZeroRotation(self.atoms)
    def step(self):
        accel = (self.atoms.get_forces() /
                 self.atoms.get_masses().reshape(-1, 1))

        vel = self.atoms.get_velocities()
        Pint=-np.sum(self.atoms.get_stress(include_ideal_gas=True)[0:3])/3
        F=3*self.atoms.get_volume()*(Pint-self.P) + (6/self.Nf)*self.atoms.get_kinetic_energy()
        G= (1/self.Q)*(2*self.atoms.get_kinetic_energy()+self.W*(self.veta**2)-(self.Nf+1)*self.T)
        eta0=self.eta
        self.eta =self.eta + self.veta * self.dt + 0.5 *((F/self.W)-self.veta*self.zeta)*self.dt*self.dt
        x = np.exp(self.eta-eta0)*(self.atoms.get_positions() + vel * self.dt + \
            (accel - self.zeta * vel - (2+(3/self.Nf))*vel*self.veta) * (0.5 * self.dt ** 2))
        self.atoms.set_positions(x)
        # make half a step in velocity
        vel_half = np.exp(self.eta-eta0)*(vel + 0.5 * self.dt * (accel - self.zeta * vel- (2+(3/self.Nf))*vel*self.veta))
        self.atoms.set_velocities(vel_half)

        # make a full step in accelerations
        f = self.atoms.get_forces()
        accel = f / self.atoms.get_masses().reshape(-1, 1)
        self.zeta = self.zeta + 0.5 * self.dt * G
        self.veta=self.veta+  0.5 * self.dt *((F/self.W)-self.veta*self.zeta)
        Vol=self.atoms.get_volume()*np.exp(3*self.eta-3*eta0)
        h=Vol**(1/3)
        self.atoms.set_cell([h,h,h])
        
        Pint=-np.sum(self.atoms.get_stress(include_ideal_gas=True)[0:3])/3
        F=3*self.atoms.get_volume()*(Pint-self.P) + (6/self.Nf)*self.atoms.get_kinetic_energy()
        G= (1/self.Q)*(2*self.atoms.get_kinetic_energy()+self.W*(self.veta**2)-(self.Nf+1)*self.T)
        self.zeta = self.zeta + 0.5 * self.dt * G
        self.veta= (self.veta + 0.5*self.dt*(F/self.W))/(1 + 0.5 * self.dt * self.zeta)
        vel = (self.atoms.get_velocities() + 0.5 * self.dt * accel) / \
            (1 + 0.5 * self.dt * self.zeta + 0.5* self.dt *(2+(3/self.Nf))*self.veta)
        self.atoms.set_velocities(vel)
        #Vol=self.atoms.get_volume()*np.exp(3*self.eta-3*eta0)
        #h=Vol**(1/3)
        #self.atoms.set_cell([h,h,h])

        return f
    def run(self, steps=None):

        if steps is None:
            steps = self.num_steps

        epochs = math.ceil(steps / self.nbr_update_period)
        # number of steps in between nbr updates
        steps_per_epoch = int(steps / epochs)
        # maximum number of steps starts at `steps_per_epoch`
        # and increments after every nbr list update
        #self.max_steps = 0
        self.atoms.update_nbr_list()

        for _ in range(epochs):
            self.max_steps += steps_per_epoch
            Dynamics.run(self)
            self.atoms.update_nbr_list()

        

class NoseHooverChainsNPT_Flexible(MolecularDynamics):
    def __init__(self,
                 atoms,
                 timestep: float,
                 temperature: float,
                 external_pressure_GPa: float,
                 freq_thermostat_per_fs: float = 0.01,
                 freq_barostat_per_fs: float = 0.1,
                 num_chains: int = 10,
                 maxwell_temp: float = None,
                 trajectory: str = None,
                 logfile: str = None,
                 loginterval: int = 1,
                 max_steps: int = None,
                 nbr_update_period: int = 20,
                 append_trajectory: bool = True,
                 **kwargs):
        
        """Docstring
           Args: 
               freq_thermostat_per_fs (float): frequency of the Nose Hoover coupling 
                                               default is 0.01 
               freq_barostat_per_fs (float): frequency of the barostat coupling 
                                             default is 0.1 
               num_chains (int): number of coupled extended DoFs, 
                                 it's known that two chains are not enough
                                 default is 10
        
        """
        
        MolecularDynamics.__init__(self,
                                   atoms=atoms,
                                   timestep=timestep * units.fs,
                                   trajectory=trajectory,
                                   logfile=logfile,
                                   loginterval=loginterval,
                                   append_trajectory=append_trajectory)

        
        self.d = 3.0 # this code is for 3D simulations
        self.dt = timestep * units.fs
        self.T = temperature * units.kB
        self.Natom = len(atoms)
        
        # no overall rotation or translation
        self.N_dof = self.d * self.Natom - 6
        self.targeEkin = 0.5 * self.N_dof * self.T

        self.num_steps = max_steps
        self.n_steps = 0
        self.max_steps = 0

        self.nbr_update_period = nbr_update_period

        # initial Maxwell-Boltmann temperature for atoms
        if maxwell_temp is None:
            maxwell_temp = temperature

        MaxwellBoltzmannDistribution(self.atoms, temperature_K=maxwell_temp)
        Stationary(self.atoms)
        ZeroRotation(self.atoms)
        
        #############
        # Barostat and Thermostat parameters
        q_0 = self.N_dof * self.T * (units.fs / freq_thermostat_per_fs)**2
        q_n = self.T * (units.fs / freq_thermostat_per_fs)**2
        # thermostat mass and coord
        self.Q = 2 * np.array([q_0, *([q_n] * (num_chains-1))])
        #self.zeta = 0.0 
        self.p_zeta = np.array([0.0]*num_chains)
        # barostat mass and coord
        self.W     = ((self.N_dof + self.d) * self.T) * (units.fs / freq_barostat_per_fs)**2
        self.Wg    = ((self.N_dof + self.d) * self.T / self.d) * (units.fs / freq_barostat_per_fs)**2
        self.eps   = 0.0 
        self.veps = 0.0
        self.P_ext = external_pressure_GPa * units.GPa
        self.vg    = np.zeros(shape=(3, 3))

    def step(self):
        # current state
        x_0     = self.atoms.get_positions(wrap=True)
        forces  = self.atoms.get_forces()
        accel   = forces / self.atoms.get_masses().reshape(-1, 1)
        current_ke = self.atoms.get_kinetic_energy()
        vel     = self.atoms.get_velocities()
        V     = self.atoms.get_volume()
        P_int = -self.atoms.get_stress(include_ideal_gas=True, voigt=False)
        # accelerations
        F_eps_0 = (self.d * V * (np.trace(P_int)/3 - self.P_ext)   # only considering hydrostatic
                   + (2 * self.d / self.N_dof) * current_ke)
        F_g_0   = (V*(P_int - np.identity(3)*self.P_ext) - 
                 (V / self.d)*np.trace(P_int - np.identity(3)*self.P_ext)*np.identity(3))
        
        dpzeta_dt = np.zeros(shape=self.p_zeta.shape)
        dpzeta_dt[0]   = (2 * (current_ke - self.targeEkin) 
                          + self.W*(self.veps**2) 
                          + self.Wg*np.trace(np.matmul(self.vg, self.vg.T))
                          - (self.d**2) * self.T 
                          - self.p_zeta[0]*self.p_zeta[1]/self.Q[1])
        dpzeta_dt[1:-1]= (np.power(self.p_zeta[:-2], 2) / self.Q[:-2] - self.T) - \
                          self.p_zeta[1:-1] * self.p_zeta[2:] / self.Q[2:]
        dpzeta_dt[-1]  = np.power(self.p_zeta[-2], 2) / self.Q[-2] - self.T
        
        delta_eps = (self.veps * self.dt 
                     + 0.5 * (F_eps_0 / self.W - self.veps * self.p_zeta[0]/self.Q[0]) * self.dt**2
                    )
        self.eps += delta_eps
        h_0 = self.atoms.get_cell()
        h_t = h_0 * (np.identity(3) + self.vg*self.dt
                     + 0.5 * (F_g_0 / self.Wg 
                              + np.matmul(self.vg.T, self.vg) # not sure whether this is the square or matrix square! 
                              - self.vg*self.p_zeta[0]/self.Q[0])* self.dt**2
                     )
        
        # half step in time
        coupl_accel = (accel 
                       - vel*self.p_zeta[0]/self.Q[0]
                       - 2.0 * np.matmul(self.vg, vel.T).T
                       - (2. + self.d / self.N_dof)*vel*self.veps)
        vel_half = np.exp(delta_eps) * np.matmul(
                          np.matmul(h_t, np.linalg.inv(h_0)), (vel + 0.5 * self.dt * coupl_accel).T
                   ).T
        
        # full step in positions
        x = np.exp(delta_eps) * np.matmul(
                np.matmul(h_t, np.linalg.inv(h_0)), (x_0 + (vel + 0.5 * self.dt * coupl_accel) * self.dt).T
                ).T
              
        self.atoms.set_velocities(vel_half)                             
        self.atoms.set_positions(x)    
        self.atoms.set_cell(h_t)
        print(self.atoms.get_cell())
        
        # half time for all velocities
        self.veps    += 0.5 * self.dt * (F_eps_0/self.W 
                                         - self.veps*self.p_zeta[0]/self.Q[0])
        self.vg      += 0.5 * self.dt * (F_g_0/self.Wg 
                                         + np.power(self.vg, 2)
                                         - self.vg*self.p_zeta[0]/self.Q[0])
        self.vg       = np.matmul(self.vg, np.matmul(h_0, np.linalg.inv(h_t)))
        self.p_zeta  += 0.5 * self.dt * dpzeta_dt
                   
        # current state
        forces_t     = self.atoms.get_forces()
        accel_t      = forces_t / self.atoms.get_masses().reshape(-1, 1)
        current_ke_t = self.atoms.get_kinetic_energy()
        V_t          = self.atoms.get_volume()
        P_int_t = -self.atoms.get_stress(include_ideal_gas=True, voigt=False)
        # accelerations
        F_eps_t = (self.d * V * (np.trace(P_int_t)/3 - self.P_ext) 
                   + (2 * self.d / self.N_dof) * current_ke_t)
        F_g_t   = (V*(P_int_t - np.identity(3)*self.P_ext) - 
                 (V / self.d)*np.trace(P_int_t - np.identity(3)*self.P_ext)*np.identity(3))
        
        dpzeta_dt = np.zeros(shape=self.p_zeta.shape)
        dpzeta_dt[0]   = (2 * (current_ke_t - self.targeEkin) 
                          + self.W*(self.veps**2) 
                          + self.Wg*np.trace(np.matmul(self.vg, self.vg.T))
                          - (self.d**2) * self.T 
                          - self.p_zeta[0]*self.p_zeta[1]/self.Q[1])
        dpzeta_dt[1:-1]= (np.power(self.p_zeta[:-2], 2) / self.Q[:-2] - self.T) - \
                          self.p_zeta[1:-1] * self.p_zeta[2:] / self.Q[2:]
        dpzeta_dt[-1]  = np.power(self.p_zeta[-2], 2) / self.Q[-2] - self.T

        # another half step in all velocities
        coupl_accel   = (accel 
                         - vel*self.p_zeta[0]/self.Q[0]
                         - 2.0 * np.matmul(self.vg, vel.T).T
                         - (2. + self.d / self.N_dof)*vel*self.veps)
        final_vel     = vel_half + 0.5 * self.dt * coupl_accel
        self.vg      += 0.5 * self.dt * (F_g_t/self.Wg 
                                         + np.matmul(self.vg.T, self.vg)
                                         - self.vg*self.p_zeta[0]/self.Q[0])
        self.veps    += 0.5 * self.dt * (F_eps_t/self.W 
                                         - self.veps*self.p_zeta[0]/self.Q[0])
        self.p_zeta  += 0.5 * self.dt * dpzeta_dt
        
        self.atoms.set_velocities(final_vel)
    
    def run(self, steps=None):

        if steps is None:
            steps = self.num_steps

        epochs = math.ceil(steps / self.nbr_update_period)
        # number of steps in between nbr updates
        steps_per_epoch = int(steps / epochs)
        # maximum number of steps starts at `steps_per_epoch`
        # and increments after every nbr list update
        #self.max_steps = 0
        self.atoms.update_nbr_list()

        for _ in tqdm(range(epochs)):
            self.max_steps += steps_per_epoch
            Dynamics.run(self)
            self.atoms.update_nbr_list()