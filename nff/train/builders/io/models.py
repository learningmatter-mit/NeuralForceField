from nff.nn.models.chemprop3d import ChemProp3D

PARAMS_TYPE = {
                "ChemProp3D":
                {
                   'n_atom_basis': int,
                   'n_filters': int,
                   'n_gaussians': int,
                   'n_convolutions': int,
                   'cutoff': float,
                   'trainable_gauss': bool,
                   'dropout_rate': float,
                   'readoutdict': dict,
                   'mol_fp_layers': list
                }
}

MODEL_DICT = {"ChemProp3D": ChemProp3D}
