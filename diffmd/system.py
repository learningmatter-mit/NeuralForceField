import torch
from ase import Atoms 
from ase import units
import numpy as np 
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution 


class System(Atoms):
    """Object that contains system information. Inherited from ase.Atoms
    
    Attributes:
        device (int or str): torch device "cpu" or an integer for GPU
        dim (int): dimension of the system cell
        props (dict{}): additional properties 
    """

    def __init__(self, *args, device, props={}, **kwargs):
        """Init of System object
        
        Args:
            device (str): torch device "cpu" or an integer for GPU
            props(dict): dictionary of additional properties
            *args and **kwargs will be passed to ase.Atoms
            
        Returns:
            -
            
        Raises:
            RuntimeError: if device variable doesn't work
            TypeError: if props is not a dictionary 
        
        """
        
        super().__init__(*args, **kwargs)
        self.props = props
        self.device = device
        self.dim = self.get_cell().shape[0]
        
        try:
            torch.Tensor([0.0]).to(device)
        except RuntimeError:
            print("device variable not set correctly")
            
        if type(props) != dict:
            raise TypeError("props is not a dictionary.")
            
        
    def set_temperature(self, T_in_K = 1.):
        MaxwellBoltzmannDistribution(self, temperature_K=T_in_K)
    