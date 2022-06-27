import torch
import numpy as np 
import math 
from ase import units
from ase.geometry import wrap_positions

from diffmd.solvers import odeint, odeint_adjoint


class MD_wrapper():
    """Wrapper object for runnindg the MD and logging
    
    Attributes:
        device (str or int): int for GPU, "cpu" for cpu
        diffeq (nn.module): function that yields acceleration and velocoties
        keys (list): name of the state variables e.g. "velocities" and "positions "
        log (dict): save state vaiables in numpy arrays at the end of a run
        method (str): integration method, current options are "Verlet" amd "NH_Verlet"
        system (diffmd.system): System object to contain state of molecular systems 
        wrap (bool): if True, wrap the coordinates based on system.cell 
    """
    
    def __init__(self, system=None, diffeq=None, method=None, wrap=False):
        """Init for the MD wrapper class
        
        Args:
            system: ase.Atom type object, contains atom types, positions, masses, etc.
            diffeq: torch class that provides the differential equation to be integrated
            method: the solver needed to integrate diffeq 
            # can method and diffeq be combined? They kinda only go together
            wrap (bool): whether the coordinates of the ase.Atoms objects will be PBC wrapped
            
        Returns:
            -
            
        Raises:
            TypeError: if system is not of type diffmd.system.System
            TypeError: if diffeq is not of type torch.nn.Modules
            TypeError: if method is not of type torch.nn.Modules
        """
        
        #JD: how do these work? they need to be improved
        #print((type(system))
        #if not isinstance(system, system.System): # != diffmd.system.Atoms_extended:
        #    raise TypeError("system does not have type diffmd.system.System!")
        
        #if type(diffeq) != torch.nn.Modules:
        #    raise TypeError("diffeq does not have type torch.nn.Modules!")
            
        #if type(method) != torch.nn.Modules:
        #    raise TypeError("method does not have type torch.nn.Modules!")
            
        self.system = system
        self.device = system.device
        self.diffeq = diffeq
        self.method = method
        self.keys = self.diffeq.state_keys
        self.initialize_log()
        self.wrap = wrap

    def initialize_log(self):
        """Initializes the log, a dictionary of lists which contain the MD trajectory

        Args:
            -

        Returns:
            -

        Raises:
            TypeError: if keys are not strings
            ValueError: if keys does not include 'velocities' and 'positions'

        """
        self.log = {}
        contains_v_and_p = 0
        for key in self.keys:
            if type(key) != str:
                raise TypeError("The keys are not strings!")
            if key == 'velocities' or key == 'positions':
                contains_v_and_p +=1
            self.log[key] = []

        if contains_v_and_p < 2:
            raise ValueError("Keys are missing at least one: velocities or positions")


    def update_log(self, traj):
        """Appends current state to the log-trajectory 

        Args:
            traj: list of torch tensor trajectories of the different state variables

        Returns:
            -

        Raises:
            -

        """
        for i, key in enumerate(self.keys):
            if traj[i][0].device != 'cpu':
                self.log[key].append(traj[i][-1].detach().cpu().numpy()) 
            else:
                self.log[key].append(traj[i][-1].detach().numpy()) 

    def update_state(self):
        """Updates the state vector of the system class for the next epoch.

        Args:
            -

        Returns:
            -

        Raises:
            -

        """

        if "positions" in self.log.keys():
            self.system.set_positions(self.log['positions'][-1])
        if "velocities" in self.log.keys():
            self.system.set_velocities(self.log['velocities'][-1])


    def get_check_point(self):
        """Gets the state vectors from the log 

        Args:
            -

        Returns:
            state: tuple of state vectors for one time step
                    state[0] = velocities
                    state[1] = positions (coordinates)
                    state[x] = additional DoFs

        Raises:
            ValueError: if no log is available

        """

        if hasattr(self, 'log'):
            state = [torch.Tensor(self.log[key][-1]).to(self.device) for key in self.log]

            if self.wrap:
                wrapped_xyz = wrap_positions(self.log['positions'][-1], self.system.get_cell())
                state[1] = torch.Tensor(wrapped_xyz).to(self.device)

            return state 
        else:
            raise ValueError("No log available")


    def simulate(self, steps=1, dt_fs=1.0):
        """Function that calls the combination of ODEs and ODE-solvers to propagate system forward in time

        Args:
            steps (int): Number of MD steps
            dt_fs (float): time step in femtoseconds

        Returns:
            traj: list of the trajectory of state variables

        Raises:
            TypeError: if steps or dt_fs have the wrong type
            ValueError: if input parameters are out of bounds

        """

        if type(steps) != int:
            raise TypeError("steps must be integer!")

        if type(dt_fs) != float:
            raise TypeError("dt_fs must be a float!")

        if steps < 1:
            raise ValueError("steps must be greater than 0!")

        if dt_fs < 0. or dt_fs > 2.:
            raise ValueError("A time step should be positive and not be greater than 2 fs!")

        if self.log['positions'] == []:
            state = self.diffeq.get_initial_state(self.wrap)
        else:
            state = self.get_check_point()

        dt = dt_fs * units.fs
        time_line = torch.Tensor([dt * i for i in range(steps)]).to(self.device)

        if self.diffeq.adjoint:
            traj = odeint_adjoint(self.diffeq, state, time_line, method=self.method)
        else:
            for variable in state:
                variable.requires_grad = True 
            traj = odeint(self.diffeq, tuple(state), time_line, method=self.method)
        self.update_log(traj)
        self.update_state()

        #state = self.get_check_point()

        return traj 