import os
import numpy as np
import torch
from typing import Union, Tuple

from ase import Atoms
from ase.neighborlist import neighbor_list
from ase.calculators.calculator import Calculator, all_changes
from ase import units

import nff.utils.constants as const
from nff.nn.utils import torch_nbr_list
from nff.utils.cuda import batch_to
from nff.data.sparse import sparsify_array
from nff.train.builders.model import load_model
from nff.utils.geom import compute_distances, batch_compute_distance
from nff.utils.scatter import compute_grad
from nff.data import Dataset
from nff.nn.graphop import split_and_sum

from nff.io.ase import NeuralFF

from nff.nn.models.schnet import SchNet, SchNetDiabat
from nff.nn.models.hybridgraph import HybridGraphConv
from nff.nn.models.schnet_features import SchNetFeatures
from nff.nn.models.cp3d import OnlyBondUpdateCP3D

from nff.data import collate_dicts

DEFAULT_CUTOFF = 5.0
DEFAULT_DIRECTED = False
DEFAULT_SKIN = 1.0
UNDIRECTED = [SchNet,
              SchNetDiabat,
              HybridGraphConv,
              SchNetFeatures,
              OnlyBondUpdateCP3D]


class BiasBase(NeuralFF):
    """Basic Calculator class with neural force field
    
    Args:
        model: the deural force field model
        cv_def: lsit of Collective Variable (CV) definitions
            [["cv_type", [atom_indices], np.array([minimum, maximum]), bin_width], [possible second dimension]]
        confinement_k: force constant for confinement of system to the range of interest in CV space
    """
    
    def __init__(self,
                 model,
                 cv_defs: list[dict],
                 equil_temp: float = 300.0,
                 device='cpu',
                 en_key='energy',
                 directed=DEFAULT_DIRECTED,
                 **kwargs):

        NeuralFF.__init__(self,
                          model=model,
                          device=device,
                          en_key=en_key,
                          directed=DEFAULT_DIRECTED,
                          funcvparams=None,
                          **kwargs)
        
        self.cv_defs = cv_defs
        self.num_cv = len(cv_defs)
        self.the_cv = []
        for cv_def in self.cv_defs:
            self.the_cv.append(CV(cv_def["definition"]))
        
        self.equil_temp = equil_temp
        
        self.ext_coords  = np.zeros(shape=(self.num_cv,))
        self.ext_masses  = np.zeros(shape=(self.num_cv,))
        self.ext_forces  = np.zeros(shape=(self.num_cv,))
        self.ext_momenta = np.zeros(shape=(self.num_cv,))
        self.ext_k = np.zeros(shape=(self.num_cv,))
        
        self.ranges  = np.zeros(shape=(self.num_cv,2))
        self.margins = np.zeros(shape=(self.num_cv,))
        self.conf_k  = np.zeros(shape=(self.num_cv,))
        
        for ii, cv in enumerate(self.cv_defs):
            if 'range' in cv.keys():
                self.ext_coords[ii] = cv['range'][0]
                self.ranges[ii] = cv['range']
            else:
                raise PropertyNotPresent('range')
                
            if 'margin' in cv.keys():
                self.margins[ii] = cv['margin']
                
            if 'conf_k' in cv.keys():
                self.conf_k[ii] = cv['conf_k']
                
            if 'ext_k' in cv.keys():
                self.ext_k[ii] = cv['ext_k']
            elif 'ext_sigma' in cv.keys()
                self.ext_k[ii] = (units.kB * self.equil_temp) / (
                                  cv['ext_sigma'] * cv['ext_sigma'])
            else:
                raise PropertyNotPresent('ext_k/ext_sigma')
                
            if 'mass' in cv.keys():
                self.ext_masses[ii] = cv['mass']
                
        # initialize extended system at target temp of MD simulation
        for i in range(self.num_cv):
            self.ext_momenta[i] = random.gauss(0.0, 1.0) * np.sqrt(
                                    self.equil_temp * self.ext_mass[i])
        ttt = (np.power(self.ext_momenta, 2) / self.ext_mass).sum()
        ttt /= self.num_cv
        self.ext_momenta *= np.sqrt(self.equil_temp / ttt)
            
        
    @abstractmethod
    def _update_bias(self):
        pass
    
    @abstractmethod
    def _propagate_ext(self):
        pass
    
    @abstractmethod
    def _up_extmom(self):
        pass
    
    def diff(self, 
             a: Union[np.ndarray, float], 
             b: Union[np.ndarray, float], 
             cv_type: str
             ) -> Union[np.ndarray, float]:
        """get difference of elements of numbers or arrays
        in range(-inf, inf) if is_angle is False or in range(-pi, pi) if is_angle is True
        Args:
            a: number or array
            b: number or array
        Returns:
            diff: element-wise difference (a-b)
        """
        diff = a - b

        # wrap to range(-pi,pi) for angle
        if isinstance(diff, np.ndarray) and cv_type == "angle":

            diff[diff > np.pi] -= 2 * np.pi
            diff[diff < -np.pi] += 2 * np.pi

        elif cv_type == "angle":

            if diff < -np.pi:
                diff += 2 * np.pi
            elif diff > np.pi:
                diff -= 2 * np.pi

        return diff
    
    @abstractmethod
    def step_bias(self, 
                 xi: np.ndarray,
                 grad_xi: np.ndarray,
                ) -> Tuple[np.ndarray, np.ndarray]: 
        """energy and gradient of bias
        
        Args:
            curr_cv: current value of the cv
            cv_index: for multidimensional FES
            
        Returns:
            bias_ener: bias energy
            bias_grad: gradiant of the bias in CV space, needs to be dotted with the cv_gradient
        """
        
        self._propagate_ext()
        
        bias_grad = np.zeros_like(self.atoms.get_positions())
        bias_ener = 0.0        
        
        for i in range(self.ncoords):
            # harmonic coupling of extended coordinate to reaction coordinate

            dxi = self.diff(self.ext_coords[i], xi[i], self.cv_defs[i]['definition']['type'])
            self.ext_forces[i] = self.ext_k[i] * dxi
            bias_grad += self.ext_k[i] * dxi * grad_xi[i]
            bias_ener += 0.5 * self.ext_k[i] * dxi**2

            # harmonic walls for confinement to range of interest
            if self.ext_coords[i] > (self.ranges[i][1] - self.margin[i]):
                r = diff(self.ranges[i][1] - self.margin[i], self.ext_coords[i], self.cv_defs[i]['definition']['type'])
                self.ext_forces[i] -= self.conf_k[i] * r

            elif self.ext_coords[i] < (self.ranges[i][0] + self.margin[i]):
                r = diff(self.ranges[i][0] + self.margin[i], self.ext_coords[i], self.cv_defs[i]['definition']['type'])
                self.ext_forces[i] -= self.conf_k[i] * r
         
        self._update_bias()
        self._up_extmom()                
                
        return bias_ener, bias_force

    
    def calculate(
            self,
            atoms=None,
            properties=['energy', 'forces', 
                        'energy_unbiased', 'forces_unbiased', 
                        'cv_vals', 'cv_invmass', 
                        'grad_length', 'cv_grad_lengths', 'cv_dot_PES'],
            system_changes=all_changes,
    ):
        """Calculates the desired properties for the given AtomsBatch.

        Args:
        atoms (AtomsBatch): custom Atoms subclass that contains implementation
            of neighbor lists, batching and so on. Avoids the use of the Dataset
            to calculate using the models created.
        properties: list of keywords that can be present in self.results
        system_changes (default from ase)
        """

        if not any([isinstance(self.model, i) for i in UNDIRECTED]):
            check_directed(self.model, atoms)

        # for backwards compatability
        if getattr(self, "properties", None) is None:
            self.properties = properties

        Calculator.calculate(self, atoms, self.properties, system_changes)

        # run model
        batch = batch_to(atoms.get_batch(), self.device)

        # add keys so that the readout function can calculate these properties
        grad_key = self.en_key + "_grad"
        batch[self.en_key] = []
        batch[grad_key] = []

        kwargs = {}
        requires_stress = "stress" in self.properties
        if requires_stress:
            kwargs["requires_stress"] = True
        if getattr(self, "model_kwargs", None) is not None:
            kwargs.update(self.model_kwargs)

        prediction = self.model(batch, **kwargs)

        # change energy and force to numpy array and eV
        model_energy = (prediction[self.en_key].detach()
                  .cpu().numpy() * (1 / const.EV_TO_KCAL_MOL))
        
        if grad_key in prediction:
            model_grad = (prediction[grad_key].detach()
                           .cpu().numpy() * (1 / const.EV_TO_KCAL_MOL))
        else:
            raise PropertyNotPresent(grad_key)
        
        energy = copy.deepcopy(model_energy)
        grad = copy.deepcopy(model_grad)
        
        masses = atoms.get_masses()
        M_inv = 1. / np.diag(np.array([masses, masses, masses]).flatten())
        print(self.M_inv)
        
        cvs = np.zeros(shape=(self.num_cv,))
        cv_grads = np.zeros(shape=(self.num_cv, atoms.get_positions().shape))
        cv_grad_lens = np.zeros(shape=(self.num_cv,))
        cv_invmass = np.zeros(shape=(self.num_cv,))
        cv_dot_PES = np.zeros(shape=(self.num_cv,))
        for ii, cv_def in enumerate(self.cv_defs):
            xi, xi_grad = self.the_cv[ii](batch)
            cvs[ii] = xi
            cv_grads[ii] = xi_grad
            cv_grad_lens[ii] = np.linalg.norm(xi_grad.flatten())
            cv_invmass[ii] = np.matmul(xi_grad.flatten(), np.matmul(M_inv, xi_grad.flatten()))
            cv_dot_PES[ii] = np.dot(xi_grad.flatten(), model_grad.flatten())
            
        bias_ener, bias_grad = self.step_bias(xi, xi_grad)
        energy += bias_ener
        grad += bias_grad
            

        self.results = {
            'energy': energy.reshape(-1),
            'forces': -grad.reshape(-1, 3),
            'energy_unbiased': model_energy.reshape(-1),
            'forces_unbiased': -model_grad.reshape(-1, 3),
            'grad_length': np.linalg.norm(model_grad),
            'cv_vals': cvs,
            'cv_grad_lengths': cv_grads,
            'cv_invmass': cv_invmass,
            'cv_dot_PES': cv_dot_PES,
            'ext_pos': self.ext_coords,
        }

        if requires_stress:
            stress = (prediction['stress_volume'].detach()
                      .cpu().numpy() * (1 / const.EV_TO_KCAL_MOL))
            self.results['stress'] = stress * (1 / atoms.get_volume())
    