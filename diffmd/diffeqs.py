import torch
from nff.utils.scatter import compute_grad
import numpy as np 
from ase import units

from nff.data.graphs import get_neighbor_list
from nff.train import batch_to, batch_detach

"""
Diffeq classes need to have the functions:
    - forward(state): returns time derivative of all state variables
    - get_initial_state: returns state from system object
"""


class NVE(torch.nn.Module):
    """Equation of state for constant energy integrator (NVE ensemble)
    
    Attributes:
        adjoint (bool): if True using adjoint sensitivity 
        dim (int): system dimensions 
        mass (torch.Tensor): masses of each particle
        model (nn.module): (stack of) energy functions that takes coordinates as input
        N_dof (int): total number of degree of freedoms
        state_keys (list): keys of state variables "positions", "velocity" etc. 
        system (diffmd.system): system object
    """
    
    def __init__(self, potentials, system, adjoint=True, cutoff=5.0, undirected=True):
        super().__init__()
        self.model = potentials 
        self.system = system
        self.mass = torch.Tensor(system.get_masses()).to(self.system.device)
        self.dim = system.dim
        self.num_atoms = torch.LongTensor([self.mass.shape[0]]) 
        self.N_dof = (self.num_atoms * self.dim).item()
        self.adjoint = adjoint
        self.state_keys = ['velocities', 'positions']
        self.undirected = undirected
        self.cutoff = cutoff
        
    def forward(self, state):
        """ODE for NVE dynamics, yields time derivative of state variables
        
        Args:
            state (tuple): velocities and coordinate arrays
            
        Returns:
            tuple of acceleration and velocity
            
        Raises:
            -
        
        """
        with torch.set_grad_enabled(True):        
            
            v = state[0]
            q = state[1]
            
            if self.adjoint:
                q.requires_grad = True
            
            atomic_numbers = torch.from_numpy(self.system.get_atomic_numbers()).view(-1,1).to(self.system.device)
            nxyz = torch.cat((atomic_numbers, q), axis=1).type(torch.FloatTensor).to(self.system.device)
            nbr_list = get_neighbor_list(q,
                             cutoff=self.cutoff,
                             undirected=self.undirected)
            batch = {"nxyz": nxyz,
                'num_atoms': self.num_atoms,
                'nbr_list': nbr_list,
                'energy': [0.0],
                'energy_grad': torch.zeros(self.num_atoms, 3)}
            #u = self.model(q)
            #f = -compute_grad(inputs=q, output=u.sum(-1))
            results = self.model(batch)
            f = -results['energy_grad']
            dvdt = f / self.mass[:, None]

        return (dvdt, v)

    def get_initial_state(self, wrap=True):
        """Returns the state vector from the system object.
        
        Args:
            wrap (bool): whether positions should be wrapped wrt cell parameters
            
        Returns:
            state (tuple): tuple of state vectors (vels and coords)
           
        Raises:
            -
            
        """
        state = [self.system.get_velocities(), 
                self.system.get_positions(wrap=wrap)]

        state = [torch.Tensor(statevec).to(self.system.device) for statevec in state]

        return state
    
    
    
class NoseHooverChains(torch.nn.Module):
    """ODE for Nose Hoover Chains 

        Nosé, S. A unified formulation of the constant temperature molecular dynamics methods. JChemPhys 81, 511–519 (1984).
        Martyna, G. A., Klein, M. L., Tuckerman, M. Nosé-Hoover chains: The canonical ensemble via continuous dynamics. JChemPhys 97, 2635 (1992)
    
    Attributes:
        adjoint (str): if True using adjoint sensitivity 
        dim (int): system dimensions
        kT (float): kB*T
        mass (torch.Tensor): masses of each particle
        model (nn.module): (stack of) energy functions that takes coordinates as input
        N_dof (int): total number of degrees of freedom
        state_keys (list): keys of state variables "positions", "velocity" etc. 
        system (diffmd.System): system object
        num_chains (int): number of chains 
        ttime (float): multiple of time step, interaction frequency between heat bath and the system (indirectly defines the bath mass), decent choice can be 20*dt
        target_ke (float): target Kinetic energy 
    """
    
    def __init__(self, potentials, system, T_in_K, ttime, num_chains=2, adjoint=True, cutoff=5.0, undirected=True):
        super().__init__()
        self.model      = potentials 
        self.system     = system
        self.mass       = torch.Tensor(system.get_masses()).to(self.system.device)
        self.kT         = T_in_K * units.kB
        self.N_dof      = self.mass.shape[0] * system.dim
        self.target_ke  = (0.5 * self.N_dof * self.kT)
        
        self.num_chains = num_chains
        self.ttime      = ttime
        self.Q          = 2 * np.array([self.N_dof * self.kT * self.ttime**2,
                           *[self.kT * self.ttime**2]*(num_chains-1)])
        self.Q          = torch.Tensor(self.Q).to(self.system.device)
       
        self.num_atoms  = torch.LongTensor([self.mass.shape[0]])
        self.dim        = system.dim
        self.adjoint    = adjoint
        self.state_keys = ['velocities', 'positions', 'baths']
        self.undirected = undirected
        self.cutoff = cutoff

        
    def forward(self, state):
        """ODE for NVT Nosé-Hoover chains dynamics, yields time derivative of state variables
        
        Args:
            state (tuple): velocities and coordinate list
            
        Returns:
            tuple of acceleration, velocity, and dpeta_dt
            
        Raises:
            -
        
        """
        with torch.set_grad_enabled(True):        
            
            v     = state[0]
            q     = state[1]
            p_eta = state[2]
            
            if self.adjoint:
                q.requires_grad = True
            
            p = v * self.mass[:, None]
            current_ke = 0.5 * (p.pow(2) / self.mass[:, None]).sum() 
            
            #u = self.model(q)
            #f = -compute_grad(inputs=q, output=u.sum(-1))

            atomic_numbers = torch.from_numpy(self.system.get_atomic_numbers()).view(-1,1).to(self.system.device)
            nxyz = torch.cat((atomic_numbers, q), axis=1).type(torch.FloatTensor).to(self.system.device)
            nbr_list = get_neighbor_list(q,
                             cutoff=self.cutoff,
                             undirected=self.undirected)
            batch = {"nxyz": nxyz,
                'num_atoms': self.num_atoms,
                'nbr_list': nbr_list,
                'energy': [0.0],
                'energy_grad': torch.zeros(self.num_atoms, 3)}

            #u = self.model(q)
            #f = -compute_grad(inputs=q, output=u.sum(-1))
            results = self.model(batch)
            f = -results['energy_grad']

            coupled_forces = (p_eta[0] * p.reshape(-1) / self.Q[0]).reshape(-1, 3)

            dvdt = (f - coupled_forces)/self.mass[:, None]

            dpeta_dt_first = 2 * (current_ke - self.target_ke) - p_eta[0]*p_eta[1]/self.Q[1]
            dpeta_dt_mid   = (p_eta[:-2].pow(2)/self.Q[:-2] - self.kT) - p_eta[1:-1]*p_eta[2:]/self.Q[2:]
            dpeta_dt_last  = p_eta[-2].pow(2)/self.Q[-2] - self.kT
            
        return (dvdt, v, torch.cat((dpeta_dt_first[None], dpeta_dt_mid, dpeta_dt_last[None])))

    def get_initial_state(self, wrap=True):
        """Returns the state vector from the system object.
        
        Args:
            wrap (bool): whether positions should be wrapped wrt cell parameters
            
        Returns:
            state (tuple): tuple of state vectors (vels, coords, and chains)
           
        Raises:
            -
            
        """
        state = [
                self.system.get_velocities(), 
                self.system.get_positions(wrap=wrap), 
                [0.0] * self.num_chains]

        state = [torch.Tensor(statevec).to(self.system.device) for statevec in state]
        return state
