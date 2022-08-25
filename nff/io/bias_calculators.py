import os
import numpy as np
import torch
from typing import Union, Tuple
import copy

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

from nff.io.ase import NeuralFF, AtomsBatch, check_directed
from nff.data import collate_dicts
from nff.md.colvars import ColVar as CV

from nff.nn.models.schnet import SchNet, SchNetDiabat
from nff.nn.models.hybridgraph import HybridGraphConv
from nff.nn.models.schnet_features import SchNetFeatures
from nff.nn.models.cp3d import OnlyBondUpdateCP3D

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
    
    implemented_properties = ['energy', 'forces', 'stress',
                              'energy_unbiased', 'forces_unbiased', 
                              'cv_vals', 'ext_pos', 'cv_invmass', 
                              'grad_length', 'cv_grad_lengths', 
                              'cv_dot_PES']
    
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
                          directed=directed,
                          **kwargs)
        
        self.cv_defs = cv_defs
        self.num_cv = len(cv_defs)
        self.the_cv = []
        for cv_def in self.cv_defs:
            self.the_cv.append(CV(cv_def["definition"]))
        
        self.equil_temp = equil_temp
        
        self.ext_coords   = np.zeros(shape=(self.num_cv,1))
        self.ext_masses   = np.zeros(shape=(self.num_cv,1))
        self.ext_forces   = np.zeros(shape=(self.num_cv,1))
        self.ext_vel      = np.zeros(shape=(self.num_cv,1))
        self.ext_binwidth = np.zeros(shape=(self.num_cv,1))
        self.ext_k  = np.zeros(shape=(self.num_cv,))
        self.ext_dt = 0.0
        
        self.ranges  = np.zeros(shape=(self.num_cv,2))
        self.margins = np.zeros(shape=(self.num_cv,1))
        self.conf_k  = np.zeros(shape=(self.num_cv,1))
        
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
            elif 'ext_sigma' in cv.keys():
                self.ext_k[ii] = (units.kB * self.equil_temp) / (
                                  cv['ext_sigma'] * cv['ext_sigma'])
            else:
                raise PropertyNotPresent('ext_k/ext_sigma')

                
            if 'type' not in cv.keys():
                self.cv_defs[ii]['type'] = 'not_angle'
            else:
                self.cv_defs[ii]['type'] = cv['type']
                
        
    def _update_bias(self, xi: np.ndarray):
        pass
    
    def _propagate_ext(self):
        pass
    
    def _up_extvel(self):
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
        
        bias_grad = np.zeros_like(grad_xi[0])
        bias_ener = 0.0        
        
        for i in range(self.num_cv):
            # harmonic coupling of extended coordinate to reaction coordinate
            dxi = self.diff(xi[i], self.ext_coords[i], self.cv_defs[i]['type'])
            self.ext_forces[i] = self.ext_k[i] * dxi
            bias_grad += self.ext_k[i] * dxi * grad_xi[i]
            bias_ener += 0.5 * self.ext_k[i] * dxi**2

            # harmonic walls for confinement to range of interest
            if self.ext_coords[i] > (self.ranges[i][1] - self.margins[i]):
                r = self.diff(self.ranges[i][1] - self.margins[i], self.ext_coords[i], self.cv_defs[i]['type'])
                self.ext_forces[i] -= self.conf_k[i] * r

            elif self.ext_coords[i] < (self.ranges[i][0] + self.margins[i]):
                r = self.diff(self.ranges[i][0] + self.margins[i], self.ext_coords[i], self.cv_defs[i]['type'])
                self.ext_forces[i] -= self.conf_k[i] * r
         
        self._update_bias(xi)
        self._up_extvel()                
                
        return bias_ener, bias_grad

    
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
        grad   = copy.deepcopy(model_grad)
        
        inv_masses = 1. / atoms.get_masses()
        M_inv  = np.diag(np.repeat(inv_masses, 3).flatten())
        
        cvs = np.zeros(shape=(self.num_cv,1))
        cv_grads     = np.zeros(shape=(self.num_cv, 
                                   atoms.get_positions().shape[0], 
                                   atoms.get_positions().shape[1]))
        cv_grad_lens = np.zeros(shape=(self.num_cv,1))
        cv_invmass   = np.zeros(shape=(self.num_cv,1))
        cv_dot_PES   = np.zeros(shape=(self.num_cv,1))
        for ii, cv_def in enumerate(self.cv_defs):
            xi, xi_grad      = self.the_cv[ii](atoms)
            cvs[ii]          = xi
            cv_grads[ii]     = xi_grad
            cv_grad_lens[ii] = np.linalg.norm(xi_grad)
            cv_invmass[ii]   = np.matmul(xi_grad.flatten(), np.matmul(M_inv, xi_grad.flatten()))
            cv_dot_PES[ii]   = np.dot(xi_grad.flatten(), model_grad.flatten())
            
        bias_ener, bias_grad = self.step_bias(cvs, cv_grads)
        energy += bias_ener
        grad   += bias_grad
            

        self.results = {
            'energy': energy.reshape(-1),
            'forces': -grad.reshape(-1, 3),
            'energy_unbiased': model_energy.reshape(-1),
            'forces_unbiased': -model_grad.reshape(-1, 3),
            'grad_length': np.linalg.norm(model_grad),
            'cv_vals': cvs,
            'cv_grad_lengths': cv_grad_lens,
            'cv_invmass': cv_invmass,
            'cv_dot_PES': cv_dot_PES,
            'ext_pos': self.ext_coords,
        }

        if requires_stress:
            stress = (prediction['stress_volume'].detach()
                      .cpu().numpy() * (1 / const.EV_TO_KCAL_MOL))
            self.results['stress'] = stress * (1 / atoms.get_volume())
            
            
            
class eABF(BiasBase):
    """extended-system Adaptive Biasing Force Calculator 
       class with neural force field
    
    Args:
        model: the deural force field model
        cv_def: lsit of Collective Variable (CV) definitions
            [["cv_type", [atom_indices], np.array([minimum, maximum]), bin_width], [possible second dimension]]
        confinement_k: force constant for confinement of system to the range of interest in CV space
    """

    def __init__(self,
                 model,
                 cv_defs: list[dict],
                 dt: float,
                 friction_per_ps: float,
                 equil_temp: float = 300.0,
                 nfull: int = 100,
                 device='cpu',
                 en_key='energy',
                 directed=DEFAULT_DIRECTED,
                 **kwargs):

        BiasBase.__init__(self,
                          cv_defs=cv_defs,
                          equil_temp=equil_temp,
                          model=model,
                          device=device,
                          en_key=en_key,
                          directed=directed,
                          **kwargs)

   
        self.ext_dt = dt * units.fs
        self.nfull  = nfull
                                 
        for ii, cv in enumerate(self.cv_defs):
            if 'bin_width' in cv.keys():
                self.ext_binwidth[ii] = cv['bin_width']
            elif 'ext_sigma' in cv.keys():
                self.ext_binwidth[ii] = cv['ext_sigma']
            else:
                raise PropertyNotPresent('bin_width')
                
            if 'ext_pos' in cv.keys():
                # set initial position
                self.ext_coords[ii] = cv['ext_pos']
            else:
                raise PropertyNotPresent('ext_pos')
                
                            
            if 'ext_mass' in cv.keys():
                self.ext_masses[ii] = cv['ext_mass']
            else:
                raise PropertyNotPresent('ext_mass')
                
        # initialize extended system at target temp of MD simulation
        for i in range(self.num_cv):
            self.ext_vel[i] = (np.random.randn() * 
                               np.sqrt(self.equil_temp * units.kB / 
                                       self.ext_masses[i]))
         
        self.friction  = friction_per_ps * 1.0e-3 / units.fs
        self.rand_push = np.sqrt(self.equil_temp * self.friction * 
                                 self.ext_dt * units.kB / (2.0e0 * self.ext_masses))
        self.prefac1   = 2.0 / (2.0 + self.friction * self.ext_dt)
        self.prefac2   = ((2.0e0 - self.friction * self.ext_dt) / 
                          (2.0e0 + self.friction * self.ext_dt))
            

        # set up all grid accumulators for ABF
        self.nbins_per_dim = np.array([1 for i in range(self.num_cv)])
        self.grid = []
        for i in range(self.num_cv):
            self.nbins_per_dim[i] = (
                int(np.ceil(np.abs(self.ranges[i,1] - self.ranges[i,0]) / 
                            self.ext_binwidth[i]))
            )
            self.grid.append(
                np.linspace(
                    self.ranges[i, 0] + self.ext_binwidth[i] / 2,
                    self.ranges[i, 1] - self.ext_binwidth[i] / 2,
                    self.nbins_per_dim[i],
                )
            )
        self.nbins = np.prod(self.nbins_per_dim)

        # accumulators and conditional averages
        self.bias = np.zeros(
            (self.num_cv, *self.nbins_per_dim), dtype=float
        )
        self.var_force = np.zeros_like(self.bias)
        self.m2_force = np.zeros_like(self.bias)
        
        self.cv_crit = np.copy(self.bias)

        self.histogram = np.zeros(
            self.nbins_per_dim, dtype=float
        )
        self.ext_hist = np.zeros_like(self.histogram)

        
                                 
    def get_index(self, xi: np.ndarray) -> tuple:
        """get list of bin indices for current position of CVs or extended variables
        Args:
            xi (np.ndarray): Current value of collective variable
        Returns:
            bin_x (list):
        """
        bin_x = np.zeros(shape=xi.shape, dtype=np.int64)
        for i in range(self.num_cv):
            bin_x[i] = int(np.floor(np.abs(xi[i] - self.ranges[i,0]) / 
                                    self.ext_binwidth[i]))
        return tuple(bin_x.reshape(1, -1)[0])
            
            
    def _update_bias(self,
                    xi: np.ndarray):
        if ((self.ext_coords <= self.ranges[:,1]).all() and 
           (self.ext_coords >= self.ranges[:,0]).all()):

            bink = self.get_index(self.ext_coords)
            self.ext_hist[bink] += 1
                                 
            # linear ramp function
            ramp = (
                1.0
                if self.ext_hist[bink] > self.nfull
                else self.ext_hist[bink] / self.nfull
            )

            for i in range(self.num_cv):

                # apply bias force on extended system
                (
                    self.bias[i][bink],
                    self.m2_force[i][bink],
                    self.var_force[i][bink],
                ) = welford_var(
                    self.ext_hist[bink],
                    self.bias[i][bink],
                    self.m2_force[i][bink],
                    self.ext_k[i] * 
                    self.diff(xi[i], self.ext_coords[i], self.cv_defs[i]['type']),
                )
                self.ext_forces -= ramp * self.bias[i][bink] 

        """                        
        Not sure how this can be dumped/printed to work with the rest
        # xi-conditioned accumulators for CZAR
        if (xi <= self.ranges[:,1]).all() and 
               (xi >= self.ranges[:,0]).all():

            bink = self.get_index(xi)
            self.histogram[bink] += 1

            for i in range(self.num_cv):
                dx = diff(self.ext_coords[i], self.grid[i][bink[i]], 
                          self.cv_defs[i]['type'])
                self.correction_czar[i][bink] += self.ext_k[i] * dx
        """
        
                                 
    def _propagate_ext(self):

        self.ext_rand_gauss = np.random.randn(len(self.ext_vel))

        self.ext_vel += self.rand_push * self.ext_rand_gauss
        self.ext_vel += 0.5e0 * self.ext_dt * self.ext_forces / self.ext_masses
        self.ext_coords += self.prefac1 * self.ext_dt * self.ext_vel 
    
                                 
    def _up_extvel(self):
                                 
        self.ext_vel *= self.prefac2
        self.ext_vel += self.rand_push * self.ext_rand_gauss                         
        self.ext_vel += 0.5e0 * self.ext_dt * self.ext_forces / self.ext_masses
    
 
    

                                 
def welford_var(
    count: float, 
    mean: float, 
    M2: float, 
    newValue: float) -> Tuple[float, float, float]:
    """On-the-fly estimate of sample variance by Welford's online algorithm
    Args:
        count: current number of samples (with new one)
        mean: current mean
        M2: helper to get variance
        newValue: new sample
    Returns:
        mean: sample mean,
        M2: sum of powers of differences from the mean
        var: sample variance
    """
    delta = newValue - mean
    mean += delta / count
    delta2 = newValue - mean
    M2 += delta * delta2
    var = M2 / count if count > 2 else 0.0
    return mean, M2, var
