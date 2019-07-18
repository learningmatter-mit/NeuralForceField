import numpy as np

from nff.train.builders import check_parameters
from nff.nn.models import Net


def get_model(params):
    params_type = {
        'n_atom_basis': int,
        'n_filters': int,
        'n_gaussians': int,
        'n_convolutions': int,
        'cutoff': float,
        'bond_par': float,
        'trainable_gauss': bool,
        'box_size': np.array
    }

    check_parameters(params_type, params)

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
