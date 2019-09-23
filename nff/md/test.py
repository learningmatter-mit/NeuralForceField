import numpy as np
import torch

from nff.io import NeuralFF, AtomsBatch
from nff.io.tests import get_ethanol
from nff.data import Dataset
import nff.utils.constants as const
from nff.train import get_model
from nff.io import ase
from nff.md.nve import *

import unittest


class TestModules(unittest.TestCase):

    def testDynamics(self):

        dataset = Dataset.from_file('../../examples/dataset.pth.tar')
        props = dataset[0]
        atoms = ase.AtomsBatch(positions=props['nxyz'][:, 1:], 
                               numbers=props['nxyz'][:, 0], 
                               props=props)
        
        # initialize models 
        params = {
            'n_atom_basis': 64,
            'n_filters': 64,
            'n_gaussians': 32,
            'n_convolutions': 2,
            'cutoff': 5.0,
            'trainable_gauss': True
        }

        model = get_model(params)

        nff_ase = NeuralFF(model=model, device='cuda:1')
        atoms.set_calculator(nff_ase)

        nve = Dynamics(atoms, DEFAULTNVEPARAMS)
        nve.run()

if __name__ == '__main__':
    unittest.main()
