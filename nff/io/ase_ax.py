import numpy as np
import torch

from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

import nff.utils.constants as const
from nff.train import load_model
from nff.data.sparse import sparsify_array
from nff.data import Dataset
from nff.nn.utils import torch_nbr_list
from nff.nn.models.schnet import SchNet, SchNetDiabat
from nff.nn.models.hybridgraph import HybridGraphConv
from nff.nn.models.schnet_features import SchNetFeatures
from nff.nn.models.cp3d import OnlyBondUpdateCP3D


DEFAULT_CUTOFF = 5.0
DEFAULT_SKIN = 1.0
DEFAULT_DIRECTED = False
CONVERSION_DIC = {"ev": 1 / const.EV_TO_KCAL_MOL,
                  "au":  const.KCAL_TO_AU["energy"]}

UNDIRECTED = [SchNet,
              SchNetDiabat,
              HybridGraphConv,
              SchNetFeatures,
              OnlyBondUpdateCP3D]


def check_directed(model, atoms):
    model_cls = model.__class__.__name__
    msg = f"{model_cls} needs a directed neighbor list"
    assert (not atoms.undirected), msg


class AtomsBatch(Atoms):
    """Class to deal with the Neural Force Field and batch several
        Atoms objects.
    """

    def __init__(
        self,
        *args,
        props=None,
        cutoff=DEFAULT_CUTOFF,
        needs_angles=False,
        undirected=(not DEFAULT_DIRECTED),
        cutoff_skin=DEFAULT_SKIN,
        **kwargs
    ):
        """

        Args:
            *args: Description
            nbr_list (None, optional): Description
            pbc_index (None, optional): Description
            cutoff (TYPE, optional): Description
            **kwargs: Description
        """
        super().__init__(*args, **kwargs)

        self.props = {} if (props is None) else props.copy()
        self.nbr_list = self.props.get('nbr_list', None)
        self.offsets = self.props.get('offsets', None)
        self.num_atoms = self.props.get('num_atoms', len(self))
        self.cutoff = cutoff
        self.cutoff_skin = cutoff_skin

        self.needs_angles = needs_angles
        self.kj_idx = self.props.get('kj_idx')
        self.ji_idx = self.props.get('ji_idx')
        self.angle_list = self.props.get('angle_list')
        self.device = self.props.get("device", 0)
        self.undirected = undirected

    def get_nxyz(self):
        """Gets the atomic number and the positions of the atoms
            inside the unit cell of the system.

        Returns:
            nxyz (np.array): atomic numbers + cartesian coordinates
                of the atoms.
        """
        nxyz = np.concatenate([
            self.get_atomic_numbers().reshape(-1, 1),
            self.get_positions().reshape(-1, 3)
        ], axis=1)

        return nxyz

    def get_batch(self, device='cpu'):
        """Uses the properties of Atoms to create a batch
            to be sent to the model.

        Returns:
            batch (dict): batch with the keys 'nxyz',
                'num_atoms', 'nbr_list' and 'offsets'
        """

        if self.nbr_list is None:  # or self.offsets is None:
            self.update_nbr_list()
            self.props['nbr_list'] = self.nbr_list
            self.props['angle_list'] = self.angle_list
            self.props['ji_idx'] = self.ji_idx
            self.props['kj_idx'] = self.kj_idx

            self.props['offsets'] = self.offsets

        self.props['nxyz'] = torch.Tensor(self.get_nxyz())
        self.props['num_atoms'] = torch.LongTensor([len(self)])

        return self.props

    def update_nbr_list(self):
        """Update neighbor list and the periodic reindexing
            for the given Atoms object.

        Args:
            cutoff (float): maximum cutoff for which atoms are
                considered interacting.

        Returns:
            nbr_list (torch.LongTensor)
            offsets (torch.Tensor)
            nxyz (torch.Tensor)
        """

        if self.needs_angles:

            dataset = Dataset({key: [val] for key, val in
                               self.props.items()}, check_props=False)
            if "nxyz" not in dataset.props:
                dataset.props["nxyz"] = [self.get_nxyz()]

            dataset.generate_neighbor_list((self.cutoff + self.cutoff_skin),
                                           undirected=self.undirected)
            dataset.generate_angle_list()

            self.ji_idx = dataset.props['ji_idx'][0]
            self.kj_idx = dataset.props['kj_idx'][0]
            self.nbr_list = dataset.props['nbr_list'][0]
            self.angle_list = dataset.props['angle_list'][0]

            nbr_list = self.nbr_list

            if any(self.pbc):
                offsets = offsets[self.nbr_list[:, 0],
                                  self.nbr_list[:, 1], :].detach().to("cpu").numpy()
            else:
                offsets = np.zeros((self.nbr_list.shape[0], 3))

        else:
            edge_from, edge_to, offsets = torch_nbr_list(self,
                                                         (self.cutoff +
                                                          self.cutoff_skin),
                                                         self.device,
                                                         directed=(not self.undirected))
            nbr_list = torch.LongTensor(np.stack([edge_from, edge_to], axis=1))
            self.nbr_list = nbr_list

        offsets = sparsify_array(offsets.dot(self.get_cell()))
        self.offsets = offsets

        return nbr_list, offsets

    def batch_properties():
        pass

    def batch_kinetic_energy():
        pass

    def batch_virial():
        pass

    @classmethod
    def from_atoms(cls,
                   atoms,
                   props=None,
                   needs_angles=False,
                   device=0,
                   **kwargs):
        instance = cls(
            atoms,
            positions=atoms.positions,
            numbers=atoms.numbers,
            props=props,
            needs_angles=needs_angles,
            **kwargs
        )

        instance.device = device
        return instance


class NeuralFF(Calculator):
    """ASE calculator using a pretrained NeuralFF model"""

    implemented_properties = ['energy', 'forces']

    def __init__(
        self,
        model,
        device='cpu',
        output_keys=['energy'],
        conversion='ev',
        dataset_props=None,
        needs_angles=False,
        model_kwargs=None,
        **kwargs
    ):
        """Creates a NeuralFF calculator.nff/io/ase.py

        Args:
            model (TYPE): Description
            device (str): device on which the calculations will be performed 
            **kwargs: Description
            model (one of nff.nn.models)
            output_keys (list): values outputted by neural network (not including gradients)
            conversion (str): conversion of output energies and forces from kcal/mol  
            dataset_props (dict): dataset.props from an initial dataset
        """

        Calculator.__init__(self, **kwargs)
        self.model = model
        self.device = device
        self.output_keys = output_keys
        self.conversion = conversion
        self.dataset_props = dataset_props
        self.needs_angles = needs_angles
        self.model_kwargs = model_kwargs

        # don't compute any gradients that aren't needed in the
        # output keys

        if getattr(model, "grad_keys", []):
            keep_keys = [key for key in model.grad_keys
                         if key.replace("_grad", "") in self.output_keys]
            if hasattr(model, "_grad_keys"):
                model._grad_keys = keep_keys
            else:
                model.grad_keys = keep_keys

    def to(self, device):
        self.device = device
        self.model.to(device)

    def calculate(
        self,
        atomsbatch=None,
        properties=['energy', 'forces'],
        system_changes=all_changes
    ):
        """Calculates the desired properties for the given AtomsBatch.

        Args:
            atoms (AtomsBatch): custom Atoms subclass that contains implementation
                of neighbor lists, batching and so on. Avoids the use of the Dataset
                to calculate using the models created.
            properties (list of str): 'energy', 'forces' or both
            system_changes (default from ase)
        """

        if not any([isinstance(self.model, i) for i in UNDIRECTED]):
            check_directed(self.model, atomsbatch)

        Calculator.calculate(self, atomsbatch, properties, system_changes)

        # run model
        # atomsbatch = AtomsBatch.from_atoms(atoms=atoms,
        #                                    props=self.dataset_props,
        #                                    needs_angles=self.needs_angles,
        #                                    device=self.device)
        batch = atomsbatch.get_batch()

        # add keys so that the readout function can calculate these properties
        for key in self.output_keys:
            batch[key] = []
            if 'forces' in properties:
                batch[key + "_grad"] = []

        kwargs = {}
        if getattr(self, "model_kwargs", None) is not None:
            kwargs.update(self.model_kwargs)

        prediction = self.model(batch, **kwargs)

        # if you're outputting more than one key (e.g. for excited states), initialize the corresponding
        # results to an empty list
        if len(self.output_keys) != 1:
            self.results["energy"] = []
        if len(self.output_keys) != 1 and 'forces' in properties:
            self.results["forces"] = []

        for key in self.output_keys:

            assert self.conversion in CONVERSION_DIC, "Unit conversion kcal/mol to {} not supported.".format(
                self.conversion)

            value = prediction[key].detach().cpu(
            ).numpy() * CONVERSION_DIC[self.conversion]

            # if you're only outputting energy, then set energy to value
            if len(self.output_keys) == 1:
                self.results["energy"] = value.reshape(-1)
            # otherwise append it to the energy list
            else:
                self.results["energy"].append(value.reshape(-1))

            if 'forces' in properties:
                value_grad = prediction[key + "_grad"].detach(
                ).cpu().numpy() * CONVERSION_DIC[self.conversion]

                if len(self.output_keys) == 1:
                    self.results["forces"] = -value_grad.reshape(-1, 3)
                else:
                    self.results["forces"].append(-value_grad.reshape(-1, 3))

        # convert any remaining lists to numpy arrays
        for key, val in self.results.items():
            self.results[key] = np.array(val)

    @classmethod
    def from_file(
        cls,
        model_path,
        device='cuda',
        output_keys=['energy'],
        conversion='ev',
        params=None,
        model_type=None,
        needs_angles=False,
        **kwargs
    ):
        model = load_model(model_path,
                           params=params,
                           model_type=model_type)
        return cls(model,
                   device,
                   output_keys,
                   conversion,
                   needs_angles=needs_angles,
                   **kwargs)
