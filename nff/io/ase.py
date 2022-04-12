import os
import numpy as np
import torch

from ase import Atoms
from ase.neighborlist import neighbor_list
from ase.calculators.calculator import Calculator, all_changes
from ase import units

import nff.utils.constants as const
from nff.nn.utils import torch_nbr_list
from nff.utils.cuda import batch_to
from nff.data.sparse import sparsify_array
from nff.train.builders.model import load_model
from nff.utils.geom import compute_distances
from nff.utils.scatter import compute_grad
from nff.data import Dataset
from nff.nn.graphop import split_and_sum

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


def check_directed(model, atoms):
    model_cls = model.__class__.__name__
    msg = f"{model_cls} needs a directed neighbor list"
    assert atoms.directed, msg


class AtomsBatch(Atoms):
    """Class to deal with the Neural Force Field and batch several
       Atoms objects.
    """

    def __init__(
            self,
            *args,
            props=None,
            cutoff=DEFAULT_CUTOFF,
            directed=DEFAULT_DIRECTED,
            requires_large_offsets=False,
            cutoff_skin=DEFAULT_SKIN,
            device=0,
            **kwargs
    ):
        """

        Args:
            *args: Description
            nbr_list (None, optional): Description
            pbc_index (None, optional): Description
            cutoff (TYPE, optional): Description
            cutoff_skin (float): extra distance added to cutoff
                            to ensure we don't miss neighbors between nbr
                            list updates.
            **kwargs: Description
        """
        super().__init__(*args, **kwargs)

        if props is None:
            props = {}

        self.props = props
        self.nbr_list = props.get('nbr_list', None)
        self.offsets = props.get('offsets', None)
        self.directed = directed
        self.num_atoms = (props.get('num_atoms',
                                    torch.LongTensor([len(self)]))
                          .reshape(-1))
        self.props['num_atoms'] = self.num_atoms
        self.cutoff = cutoff
        self.cutoff_skin = cutoff_skin
        self.device = device
        self.requires_large_offsets = requires_large_offsets

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

    def get_batch(self):
        """Uses the properties of Atoms to create a batch
           to be sent to the model.
           Returns:
              batch (dict): batch with the keys 'nxyz',
                            'num_atoms', 'nbr_list' and 'offsets'
        """

        if self.nbr_list is None or self.offsets is None:
            self.update_nbr_list()

        self.props['nbr_list'] = self.nbr_list
        self.props['offsets'] = self.offsets

        self.props['nxyz'] = torch.Tensor(self.get_nxyz())
        if self.props.get('num_atoms') is None:
            self.props['num_atoms'] = torch.LongTensor([len(self)])

        return self.props

    def get_list_atoms(self):

        if self.props.get('num_atoms') is None:
            self.props['num_atoms'] = torch.LongTensor([len(self)])

        mol_split_idx = self.props['num_atoms'].tolist()

        positions = torch.Tensor(self.get_positions())
        Z = torch.LongTensor(self.get_atomic_numbers())

        positions = list(positions.split(mol_split_idx))
        Z = list(Z.split(mol_split_idx))

        Atoms_list = []

        for i, molecule_xyz in enumerate(positions):
            Atoms_list.append(Atoms(Z[i].tolist(),
                                    molecule_xyz.numpy(),
                                    cell=self.cell,
                                    pbc=self.pbc))

        return Atoms_list

    def update_nbr_list(self):
        """Update neighbor list and the periodic reindexing
           for the given Atoms object.

           Args:
           cutoff(float): maximum cutoff for which atoms are
                                          considered interacting.
           Returns:
           nbr_list(torch.LongTensor)
           offsets(torch.Tensor)
           nxyz(torch.Tensor)

        """

        Atoms_list = self.get_list_atoms()

        ensemble_nbr_list = []
        ensemble_offsets_list = []

        for i, atoms in enumerate(Atoms_list):
            edge_from, edge_to, offsets = torch_nbr_list(
                atoms,
                (self.cutoff + self.cutoff_skin),
                device=self.device,
                directed=self.directed,
                requires_large_offsets=self.requires_large_offsets)
            nbr_list = torch.LongTensor(np.stack([edge_from, edge_to], axis=1))
            these_offsets = sparsify_array(offsets.dot(self.get_cell()))

            # non-periodic
            if isinstance(these_offsets, int):
                these_offsets = torch.Tensor(offsets)

            ensemble_nbr_list.append(
                self.props['num_atoms'][: i].sum() + nbr_list)
            ensemble_offsets_list.append(these_offsets)

        ensemble_nbr_list = torch.cat(ensemble_nbr_list)

        if all([isinstance(i, int) for i in ensemble_offsets_list]):
            ensemble_offsets_list = torch.Tensor(ensemble_offsets_list)
        else:
            ensemble_offsets_list = torch.cat(ensemble_offsets_list)

        self.nbr_list = ensemble_nbr_list
        self.offsets = ensemble_offsets_list

        return ensemble_nbr_list, ensemble_offsets_list

    def get_batch_energies(self):

        if self._calc is None:
            raise RuntimeError('Atoms object has no calculator.')

        if not hasattr(self._calc, 'get_potential_energies'):
            raise RuntimeError(
                'The calculator for atomwise energies is not implemented')

        energies = self.get_potential_energies()

        batched_energies = split_and_sum(torch.Tensor(energies),
                                         self.props['num_atoms'].tolist())

        return batched_energies.detach().cpu().numpy()

    def get_batch_kinetic_energy(self):

        if self.get_momenta().any():
            atomwise_ke = torch.Tensor(
                0.5 * self.get_momenta() * self.get_velocities()).sum(-1)
            batch_ke = split_and_sum(
                atomwise_ke, self.props['num_atoms'].tolist())
            return batch_ke.detach().cpu().numpy()

        else:
            print("No momenta are set for atoms")

    def get_batch_T(self):

        T = (self.get_batch_kinetic_energy()
             / (1.5 * units.kB * self.props['num_atoms']
                .detach().cpu().numpy()))
        return T

    def batch_properties():
        pass

    def batch_virial():
        pass

    @classmethod
    def from_atoms(cls, atoms):
        return cls(
            atoms,
            positions=atoms.positions,
            numbers=atoms.numbers,
            props={},
        )


class BulkPhaseMaterials(Atoms):
    """Class to deal with the Neural Force Field and batch molecules together
    in a box for handling boxphase.
    """

    def __init__(
            self,
            *args,
            props={},
            cutoff=DEFAULT_CUTOFF,
            nbr_torch=False,
            device='cpu',
            directed=DEFAULT_DIRECTED,
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

        self.props = props
        self.nbr_list = self.props.get('nbr_list', None)
        self.offsets = self.props.get('offsets', None)
        self.num_atoms = self.props.get('num_atoms', None)
        self.cutoff = cutoff
        self.nbr_torch = nbr_torch
        self.device = device
        self.directed = directed

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

    def get_batch(self):
        """Uses the properties of Atoms to create a batch
           to be sent to the model.

        Returns:
           batch (dict): batch with the keys 'nxyz',
           'num_atoms', 'nbr_list' and 'offsets'
        """

        if self.nbr_list is None or self.offsets is None:
            self.update_nbr_list()
            self.props['nbr_list'] = self.nbr_list
            self.props['atoms_nbr_list'] = self.atoms_nbr_list
            self.props['offsets'] = self.offsets

        self.props['nbr_list'] = self.nbr_list
        self.props['atoms_nbr_list'] = self.atoms_nbr_list
        self.props['offsets'] = self.offsets
        self.props['nxyz'] = torch.Tensor(self.get_nxyz())

        return self.props

    def update_system_nbr_list(self, cutoff, exclude_atoms_nbr_list=True):
        """Update undirected neighbor list and the periodic reindexing
           for the given Atoms object.

           Args:
           cutoff (float): maximum cutoff for which atoms are
           considered interacting.

           Returns:
           nbr_list (torch.LongTensor)
           offsets (torch.Tensor)
                nxyz (torch.Tensor)
        """

        if self.nbr_torch:
            edge_from, edge_to, offsets = torch_nbr_list(
                self, self.cutoff, device=self.device)
            nbr_list = torch.LongTensor(np.stack([edge_from, edge_to], axis=1))
        else:
            edge_from, edge_to, offsets = neighbor_list(
                'ijS',
                self,
                self.cutoff)
            nbr_list = torch.LongTensor(np.stack([edge_from, edge_to], axis=1))
            if not getattr(self, "directed", DEFAULT_DIRECTED):
                offsets = offsets[nbr_list[:, 1] > nbr_list[:, 0]]
                nbr_list = nbr_list[nbr_list[:, 1] > nbr_list[:, 0]]

        if exclude_atoms_nbr_list:
            offsets_mat = torch.zeros(self.get_number_of_atoms(),
                                      self.get_number_of_atoms(), 3)
            nbr_list_mat = (torch.zeros(self.get_number_of_atoms(),
                                        self.get_number_of_atoms())
                            .to(torch.long))
            atom_nbr_list_mat = (torch.zeros(self.get_number_of_atoms(),
                                             self.get_number_of_atoms())
                                 .to(torch.long))

            offsets_mat[nbr_list[:, 0], nbr_list[:, 1]] = offsets
            nbr_list_mat[nbr_list[:, 0], nbr_list[:, 1]] = 1
            atom_nbr_list_mat[self.atoms_nbr_list[:, 0],
                              self.atoms_nbr_list[:, 1]] = 1

            nbr_list_mat = nbr_list_mat - atom_nbr_list_mat
            nbr_list = nbr_list_mat.nonzero()
            offsets = offsets_mat[nbr_list[:, 0], nbr_list[:, 1], :]

        self.nbr_list = nbr_list
        self.offsets = sparsify_array(
            offsets.matmul(torch.Tensor(self.get_cell())))

    def get_list_atoms(self):

        mol_split_idx = self.props['num_subgraphs'].tolist()

        positions = torch.Tensor(self.get_positions())
        Z = torch.LongTensor(self.get_atomic_numbers())

        positions = list(positions.split(mol_split_idx))
        Z = list(Z.split(mol_split_idx))

        Atoms_list = []

        for i, molecule_xyz in enumerate(positions):
            Atoms_list.append(Atoms(Z[i].tolist(),
                                    molecule_xyz.numpy(),
                                    cell=self.cell,
                                    pbc=self.pbc))

        return Atoms_list

    def update_atoms_nbr_list(self, cutoff):

        Atoms_list = self.get_list_atoms()

        intra_nbr_list = []
        for i, atoms in enumerate(Atoms_list):
            edge_from, edge_to = neighbor_list('ij', atoms, cutoff)
            nbr_list = torch.LongTensor(np.stack([edge_from, edge_to], axis=1))

            if not self.directed:
                nbr_list = nbr_list[nbr_list[:, 1] > nbr_list[:, 0]]

            intra_nbr_list.append(
                self.props['num_subgraphs'][: i].sum() + nbr_list)

        intra_nbr_list = torch.cat(intra_nbr_list)
        self.atoms_nbr_list = intra_nbr_list

    def update_nbr_list(self):
        self.update_atoms_nbr_list(self.props['atoms_cutoff'])
        self.update_system_nbr_list(self.props['system_cutoff'])


class NeuralFF(Calculator):
    """ASE calculator using a pretrained NeuralFF model"""

    implemented_properties = ['energy', 'forces', 'stress']

    def __init__(
            self,
            model,
            device='cpu',
            en_key='energy',
            properties=['energy', 'forces'],
            model_kwargs=None,
            **kwargs
    ):
        """Creates a NeuralFF calculator.nff/io/ase.py

        Args:
        model (TYPE): Description
        device (str): device on which the calculations will be performed
        properties (list of str): 'energy', 'forces' or both and also stress for only
        schnet  and painn
        **kwargs: Description
        model (one of nff.nn.models)
        """

        Calculator.__init__(self, **kwargs)
        self.model = model
        self.model.eval()
        self.device = device
        self.to(device)
        self.en_key = en_key
        self.properties = properties
        self.model_kwargs = model_kwargs

    def to(self, device):
        self.device = device
        self.model.to(device)

    def calculate(
            self,
            atoms=None,
            properties=['energy', 'forces'],
            system_changes=all_changes,
    ):
        """Calculates the desired properties for the given AtomsBatch.

        Args:
        atoms (AtomsBatch): custom Atoms subclass that contains implementation
            of neighbor lists, batching and so on. Avoids the use of the Dataset
            to calculate using the models created.
        system_changes (default from ase)
        """

        if not any([isinstance(self.model, i) for i in UNDIRECTED]):
            check_directed(self.model, atoms)

        # for backwards compatability
        if getattr(self, "properties", None) is None:
            self.properties = properties

        Calculator.calculate(self, atoms, self.properties, system_changes)

        # run model
        # atomsbatch = AtomsBatch(atoms)
        # batch_to(atomsbatch.get_batch(), self.device)
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

        # change energy and force to numpy array

        energy = (prediction[self.en_key].detach()
                  .cpu().numpy() * (1 / const.EV_TO_KCAL_MOL))

        if grad_key in prediction:
            energy_grad = (prediction[grad_key].detach()
                           .cpu().numpy() * (1 / const.EV_TO_KCAL_MOL))

        self.results = {
            'energy': energy.reshape(-1)
        }

        if 'forces' in self.properties:
            self.results['forces'] = -energy_grad.reshape(-1, 3)
        if requires_stress:
            stress = (prediction['stress_volume'].detach()
                      .cpu().numpy() * (1 / const.EV_TO_KCAL_MOL))
            self.results['stress'] = stress * (1 / atoms.get_volume())

    @classmethod
    def from_file(
            cls,
            model_path,
            device='cuda',
            **kwargs
    ):

        model = load_model(model_path)
        out = cls(model=model,
                  device=device,
                  **kwargs)
        return out


class EnsembleNFF(Calculator):
    """Produces an ensemble of NFF calculators to predict the
       discrepancy between the properties"""
    implemented_properties = ['energy', 'forces']

    def __init__(
            self,
            models: list,
            device='cpu',
            **kwargs
    ):
        """Creates a NeuralFF calculator.nff/io/ase.py

        Args:
        model(TYPE): Description
        device(str): device on which the calculations will be performed
        **kwargs: Description
        model(one of nff.nn.models)

        """

        Calculator.__init__(self, **kwargs)
        self.models = models
        for m in self.models:
            m.eval()
        self.device = device
        self.to(device)

    def to(self, device):
        self.device = device
        for m in self.models:
            m.to(device)

    def calculate(
            self,
            atoms=None,
            properties=['energy', 'forces'],
            system_changes=all_changes,
    ):
        """Calculates the desired properties for the given AtomsBatch.

        Args:
        atoms (AtomsBatch): custom Atoms subclass that contains implementation
            of neighbor lists, batching and so on. Avoids the use of the Dataset
            to calculate using the models created.
        properties (list of str): 'energy', 'forces' or both
        system_changes (default from ase)
        """

        for model in self.models:
            if not any([isinstance(model, i) for i in UNDIRECTED]):
                check_directed(model, atoms)

        Calculator.calculate(self, atoms, properties, system_changes)

        # run model
        # atomsbatch = AtomsBatch(atoms)
        # batch_to(atomsbatch.get_batch(), self.device)
        batch = batch_to(atoms.get_batch(), self.device)

        # add keys so that the readout function can calculate these properties
        batch['energy'] = []
        if 'forces' in properties:
            batch['energy_grad'] = []

        energies = []
        gradients = []
        for model in self.models:
            prediction = model(batch)

            # change energy and force to numpy array
            energies.append(
                prediction['energy']
                    .detach()
                    .cpu()
                    .numpy()
                * (1 / const.EV_TO_KCAL_MOL)
            )
            gradients.append(
                prediction['energy_grad']
                .detach()
                .cpu()
                .numpy()
                * (1 / const.EV_TO_KCAL_MOL)
            )

        energies = np.stack(energies)
        gradients = np.stack(gradients)

        self.results = {
            'energy': energies.mean(0).reshape(-1),
            'energy_std': energies.std(0).reshape(-1),
        }

        if 'forces' in properties:
            self.results['forces'] = -gradients.mean(0).reshape(-1, 3)
            self.results['forces_std'] = gradients.std(0).reshape(-1, 3)

        atoms.results = self.results.copy()

    @classmethod
    def from_files(
            cls,
            model_paths: list,
            device='cuda',
            **kwargs
    ):
        models = [
            load_model(path)
            for path in model_paths
        ]
        return cls(models, device, **kwargs)


class NeuralOptimizer:
    def __init__(
            self,
            optimizer,
            nbrlist_update_freq=5
    ):
        self.optimizer = optimizer
        self.update_freq = nbrlist_update_freq

    def run(self, fmax=0.2, steps=1000):
        epochs = steps // self.update_freq

        for step in range(epochs):
            self.optimizer.run(fmax=fmax, steps=self.update_freq)
            self.optimizer.atoms.update_nbr_list()


class NeuralMetadynamics(NeuralFF):

    def __init__(self,
                 model,
                 pushing_params,
                 old_atoms=None,
                 device='cpu',
                 en_key='energy',
                 directed=DEFAULT_DIRECTED,
                 **kwargs):

        NeuralFF.__init__(self,
                          model=model,
                          device=device,
                          en_key=en_key,
                          directed=DEFAULT_DIRECTED,
                          **kwargs)

        self.pushing_params = pushing_params
        self.old_atoms = old_atoms if (old_atoms is not None) else []
        self.steps_from_old = []

        # only apply the bias to certain atoms
        self.exclude_atoms = torch.LongTensor(self.pushing_params
                                              .get("exclude_atoms", []))

    def get_keep_idx(self, atoms):
        # correct for atoms not in the biasing potential

        keep_idx = torch.LongTensor([i for i in range(len(atoms))
                                     if i not in self.exclude_atoms])
        return keep_idx

    def make_dsets(self,
                   atoms):

        keep_idx = self.get_keep_idx(atoms)
        # put the current atoms as the second dataset because that's the one
        # that gets its positions rotated and its gradient computed
        props_1 = {"nxyz": [torch.Tensor(atoms.get_nxyz())
                            [keep_idx, :]]}
        props_0 = {"nxyz": [torch.Tensor(old_atoms.get_nxyz())[keep_idx, :]
                            for old_atoms in self.old_atoms]}

        dset_0 = Dataset(props_0, do_copy=False)
        dset_1 = Dataset(props_1, do_copy=False)

        return dset_0, dset_1

    def rmsd_prelims(self, atoms):

        num_atoms = len(atoms)

        # f_damp is the turn-on on timescale, measured in
        # number of steps. From https://pubs.acs.org/doi/pdf/
        # 10.1021/acs.jctc.9b00143

        kappa = self.pushing_params["kappa"]
        steps_from_old = torch.Tensor(self.steps_from_old)
        f_damp = (2 / (1 + torch.exp(-kappa * steps_from_old)) -
                  1)

        # given in mHartree / atom in CREST paper
        k_i = ((self.pushing_params['k_i'] / 1000 *
                units.Hartree * num_atoms))

        # given in Bohr^(-2) in CREST paper
        alpha_i = ((self.pushing_params['alpha_i'] /
                    units.Bohr ** 2))

        dsets = self.make_dsets(atoms)

        return k_i, alpha_i, dsets, f_damp

    def rmsd_push(self, atoms):

        if not self.old_atoms:
            return np.zeros((len(atoms), 3)), 0.0

        k_i, alpha_i, dsets, f_damp = self.rmsd_prelims(atoms)

        delta_i, _, xyz_list = compute_distances(
            dataset=dsets[0],
            # do this on CPU - it's a small RMSD
            # and gradient calculation, so the
            # dominant time is data transfer to GPU.
            # Testing it out confirms that you get a
            # big slowdown from doing it on GPU
            device='cpu',
            # device=self.device,
            dataset_1=dsets[1],
            store_grad=True,
            collate_dicts=collate_dicts)

        v_bias = (f_damp * k_i * torch.exp(-alpha_i * delta_i.reshape(-1) ** 2)
                  ).sum()
        f_bias = -compute_grad(inputs=xyz_list[0],
                               output=v_bias).sum(0)

        keep_idx = self.get_keep_idx(atoms)
        final_f_bias = torch.zeros(len(atoms), 3)
        final_f_bias[keep_idx] = f_bias.detach().cpu()
        nan_idx = torch.bitwise_not(torch.isfinite(final_f_bias))
        final_f_bias[nan_idx] = 0

        return final_f_bias.detach().numpy(), v_bias.detach().numpy()

    def get_bias(self, atoms):
        bias_type = self.pushing_params['bias_type']
        if bias_type == "rmsd":
            results = self.rmsd_push(atoms)
        else:
            raise NotImplementedError

        return results

    def append_atoms(self, atoms):
        self.old_atoms.append(atoms)
        self.steps_from_old.append(0)

        max_ref = self.pushing_params.get("max_ref_structures")
        if max_ref is None:
            max_ref = 10

        if len(self.old_atoms) >= max_ref:
            self.old_atoms = self.old_atoms[-max_ref:]
            self.steps_from_old = self.steps_from_old[-max_ref:]

    def calculate(self,
                  atoms,
                  properties=['energy', 'forces'],
                  system_changes=all_changes,
                  add_steps=True):

        if not any([isinstance(self.model, i) for i in UNDIRECTED]):
            check_directed(self.model, atoms)

        super().calculate(atoms=atoms,
                          properties=properties,
                          system_changes=system_changes)

        # Add metadynamics energy and forces

        f_bias, _ = self.get_bias(atoms)

        self.results['forces'] += f_bias
        self.results['f_bias'] = f_bias

        if add_steps:
            for i, step in enumerate(self.steps_from_old):
                self.steps_from_old[i] = step + 1


class BatchNeuralMetadynamics(NeuralMetadynamics):

    def __init__(self,
                 model,
                 pushing_params,
                 old_atoms=None,
                 device='cpu',
                 en_key='energy',
                 directed=DEFAULT_DIRECTED,
                 **kwargs):

        NeuralMetadynamics.__init__(self,
                                    model=model,
                                    pushing_params=pushing_params,
                                    old_atoms=old_atoms,
                                    device=device,
                                    en_key=en_key,
                                    directed=directed,
                                    **kwargs)

        # import pdb
        # pdb.set_trace()

    def make_dsets(self,
                   atoms):

        dset_0s = []
        dset_1s = []

        all_idx = torch.arange(len(atoms))
        all_idx[self.exclude_atoms] = -1

        split_idx = list(torch.split(all_idx,
                                     atoms.num_atoms.tolist()))
        nxyz = atoms.get_nxyz()
        old_nxyz_list = [old_atoms.get_nxyz() for old_atoms in self.old_atoms]

        for these_idx in split_idx:
            use_idx = these_idx[these_idx != -1]

            use_nxyz = [torch.Tensor(nxyz[use_idx, :])]
            use_old_nxyz = [torch.Tensor(old_nxyz[use_idx, :])
                            for old_nxyz in old_nxyz_list]

            props_1 = {"nxyz": use_nxyz}
            props_0 = {"nxyz": use_old_nxyz}

            dset_0 = Dataset(props_0, do_copy=False)
            dset_1 = Dataset(props_1, do_copy=False)

            dset_0s.append(dset_0)
            dset_1s.append(dset_1)

        return dset_0s, dset_1s

    def rmsd_push(self, atoms):

        if not self.old_atoms:
            return np.zeros((len(atoms), 3)), np.zeros(len(atoms.num_atoms))

        _, alpha_i, dsets, f_damp = self.rmsd_prelims(atoms)
        # k_i depends on the number of atoms so must be done by batch
        k_i = ((self.pushing_params['k_i'] / 1000 *
                units.Hartree * atoms.num_atoms))

        f_biases = []
        v_biases = []

        dset0s, dset1s = dsets

        for j, dset_0 in enumerate(dset0s):

            dset_1 = dset1s[j]
            delta_i, _, xyz_list = compute_distances(
                dataset=dset_0,
                device='cpu',
                dataset_1=dset_1,
                store_grad=True,
                collate_dicts=collate_dicts)

            v_bias = (f_damp * k_i[j] * torch.exp(-alpha_i * delta_i.reshape(-1) ** 2)
                      ).sum()

            f_bias = -compute_grad(inputs=xyz_list[0],
                                   output=v_bias).sum(0).detach().cpu()
            v_bias = v_bias.detach().cpu()

            v_biases.append(v_bias.reshape(-1))
            f_biases.append(f_bias)

        v_biases = torch.cat(v_biases)
        f_biases = torch.cat(f_biases)

        keep_idx = self.get_keep_idx(atoms)
        final_f_bias = torch.zeros(len(atoms), 3)
        final_f_bias[keep_idx] = f_biases
        nan_idx = torch.bitwise_not(torch.isfinite(final_f_bias))
        final_f_bias[nan_idx] = 0

        return final_f_bias.numpy(), v_biases.numpy()


class NeuralGAMD(NeuralFF):
    """
    NeuralFF for Gaussian-accelerated molecular dynamics (GAMD)
    """

    def __init__(self,
                 model,
                 k_0,
                 V_min,
                 V_max,
                 device=0,
                 en_key='energy',
                 directed=DEFAULT_DIRECTED,
                 **kwargs):

        NeuralFF.__init__(self,
                          model=model,
                          device=device,
                          en_key=en_key,
                          directed=DEFAULT_DIRECTED,
                          **kwargs)
        self.V_min = V_min
        self.V_max = V_max

        self.k_0 = k_0
        self.k = self.k_0 / (self.V_max - self.V_min)

    def calculate(self,
                  atoms,
                  properties=['energy', 'forces'],
                  system_changes=all_changes):

        if not any([isinstance(self.model, i) for i in UNDIRECTED]):
            check_directed(self.model, atoms)

        super().calculate(atoms=atoms,
                          properties=properties,
                          system_changes=system_changes)

        old_en = self.results['energy']
        if old_en < self.V_max:
            old_forces = self.results['forces']
            f_bias = -self.k * (self.V_max - old_en) * old_forces

            self.results['forces'] += f_bias
