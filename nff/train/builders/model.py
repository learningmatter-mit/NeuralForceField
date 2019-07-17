import numpy as np

from nff.train.builder import Builder
from nff.nn.models import Net

class ModelBuilder(Builder):
    """Builder for the NFF models"""

    @property
    def params_type(self):
        return {
            'n_filters': int,
            'n_gaussians': int,
            'n_convolutions': int,
            'cutoff': float,
            'bond_par': float,
            'trainable_gauss': bool,
            'box_size': np.array
        }

    @classmethod
    def get_model(cls, params):
        cls.check_parameters(params)

        model = Net(
            n_atom_basis=params['n_atom_basis'],
            n_filters=params['n_filters'],
            n_gaussians=params['n_gaussians'], 
            n_convolutions=params['n_convolutions'],
            cutoff=params['cutoff'], 
            device=params['device'],
            bond_par=params.get('bond_par', 50.0),
            trainable_gauss=params.get('trainable_gauss', False),
            box_size=params.get('box_size', None)
        )

        return model
