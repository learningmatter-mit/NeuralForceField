import torch
from nff.utils.scatter import compute_grad
from nff.utils import batch_to
from torch.nn import ModuleDict
from ase import Atoms 
from ase import units
import numpy as np 
from nff.utils.scatter import compute_grad

import nff.utils.constants as const

class Stack(torch.nn.Module):

    """Summary
    
    Attributes:
        models (TYPE): Description
    """
    
    def __init__(self, model_dict):
        """Turns the input dictionary into a torch.nn.ModuleDict
        
        Args:
            model_dict (dictionary): dictionary of potentials
            
        Returns:
            -
            
        Raises:
            -
        """
        super().__init__()
        self.models = ModuleDict(model_dict)
        
    def forward(self, x):
        """Summary
        
        Args:
            x (torch.Tensor): coordinates oof the system
        
        Returns:
            float: Sum of the potential energy of the different PES functions
            
        Raises:
            -
        """
        for i, key in enumerate(self.models.keys()):
            if i == 0:
                #result = self.models[key](x).sum().reshape(-1)
                result = self.models[key](x)
                #print(key, result)
            else:
                #result += self.models[key](x).sum().reshape(-1)
                result_temp = self.models[key](x)
                #print(key, result_temp)
                result['energy']      += result_temp['energy']
                result['energy_grad'] += result_temp['energy_grad']
        
        result['energy']      /= const.EV_TO_KCAL_MOL
        result['energy_grad'] /= const.EV_TO_KCAL_MOL

        return result


