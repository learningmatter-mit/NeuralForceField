from nff.io.ase import AtomsBatch
from nff.io.ase_calcs import NeuralFF
from nff.data import Dataset
from nff.train import get_model
from nff.md.nve import Dynamics, DEFAULTNVEPARAMS

import unittest


class TestModules(unittest.TestCase):
    def testDynamics(self):
        dataset = Dataset.from_file("../../examples/dataset.pth.tar")
        props = dataset[0]
        atoms = AtomsBatch(
            positions=props["nxyz"][:, 1:], numbers=props["nxyz"][:, 0], props=props
        )

        # initialize models
        params = {
            "n_atom_basis": 64,
            "n_filters": 64,
            "n_gaussians": 32,
            "n_convolutions": 2,
            "cutoff": 5.0,
            "trainable_gauss": True,
        }

        model = get_model(params)

        nff_ase = NeuralFF(model=model, device="cuda:1")
        atoms.set_calculator(nff_ase)

        nve = Dynamics(atoms, DEFAULTNVEPARAMS)
        nve.run()


if __name__ == "__main__":
    unittest.main()
